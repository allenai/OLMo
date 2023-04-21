use std::io;
use std::io::{BufReader, BufWriter, Write};
use std::collections::VecDeque;
use byteorder::{LittleEndian, NativeEndian, ReadBytesExt, WriteBytesExt};
use std::sync::atomic::{AtomicU32, Ordering};
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::size_of;
use rand::Rng;
use ahash::RandomState;
use std::fs::OpenOptions;
use std::path::PathBuf;

use config::BloomFilterConfig;
use crate::config;

pub struct BloomFilter {
    bits: Vec<AtomicU32>,
    hash_builder_seeds: Vec<[u64; 4]>,
    // RandomState does not store its seeds, so we have to store them ourselves.
    hash_builders: Vec<RandomState>,
}

impl BloomFilter {
    const MAGIC: u32 = 0x81F0F117;
    const VERSION: u32 = 1;

    pub fn optimal_number_of_hashers(size_in_bytes: usize, expected_elements: usize) -> usize {
        let expected_elements = expected_elements as f64;
        let size_in_bits = (size_in_bytes * 8) as f64;
        let k = (size_in_bits / expected_elements) * (2.0f64.ln());
        k.ceil() as usize
    }

    pub fn prob_of_false_positive(size_in_bytes: usize, expected_elements: usize, num_hashers: usize) -> f64 {
        let k = num_hashers as f64;
        let m = (size_in_bytes * 8) as f64;
        let n = expected_elements as f64;
        (1.0 - (1.0 - (1.0 / m)).powf(k * n)).powf(k)
    }

    pub fn suggest_size_in_bytes(expected_elements: usize, desired_false_positive_rate: f64) -> usize {
        let mut size_in_bytes = 1024 * 1024;
        while size_in_bytes < usize::MAX / 2 && Self::prob_of_false_positive(
            size_in_bytes,
            expected_elements,
            Self::optimal_number_of_hashers(size_in_bytes, expected_elements),
        ) > desired_false_positive_rate {
            size_in_bytes *= 2;
        }
        size_in_bytes
    }

    pub fn my_prob_of_false_positive(&self, expected_elements: usize) -> f64 {
        Self::prob_of_false_positive(
            self.size_in_bytes(),
            expected_elements,
            self.hash_builders.len())
    }

    pub fn size_in_bytes(&self) -> usize {
        self.bits.len() * size_of::<AtomicU32>()
    }

    pub fn new(size_in_bytes: usize, num_hashers: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers);
        let mut hash_builders = Vec::with_capacity(num_hashers);
        for _ in 0..num_hashers {
            let seeds = rng.gen::<[u64; 4]>();
            hash_builders.push(RandomState::with_seeds(
                seeds[0],
                seeds[1],
                seeds[2],
                seeds[3]));
            hash_builder_seeds.push(seeds);
        }

        let mut bits = Vec::new();
        let number_of_u32 = size_in_bytes / size_of::<AtomicU32>();
        bits.reserve_exact(number_of_u32);
        for _ in 0..number_of_u32 {
            bits.push(AtomicU32::new(0));
        }

        Self { bits, hash_builder_seeds, hash_builders }
    }

    pub fn from_file(path: &PathBuf) -> io::Result<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(path)?;
        let mut stream = BufReader::new(&mut file);

        let magic: u32 = stream.read_u32::<LittleEndian>()?;
        if magic != Self::MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        let version: u32 = stream.read_u32::<LittleEndian>()?;
        if version != Self::VERSION {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid version"));
        }

        let num_hashers: u32 = stream.read_u32::<LittleEndian>()?;
        let mut hash_builder_seeds = Vec::with_capacity(num_hashers as usize);
        let mut hash_builders = Vec::with_capacity(num_hashers as usize);
        for _ in 0..num_hashers {
            let seeds = [
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
                stream.read_u64::<LittleEndian>()?,
            ];
            hash_builders.push(RandomState::with_seeds(
                seeds[0],
                seeds[1],
                seeds[2],
                seeds[3]));
            hash_builder_seeds.push(seeds);
        }

        let number_of_elements = stream.read_u64::<LittleEndian>()?;
        let mut bits = Vec::new();
        bits.reserve_exact(number_of_elements as usize);
        for _ in 0..number_of_elements {
            bits.push(AtomicU32::new(stream.read_u32::<NativeEndian>()?));
        }

        Ok(Self { bits, hash_builder_seeds, hash_builders })
    }

    pub fn write_to_file(&self, path: &PathBuf) -> io::Result<()> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        let mut stream = BufWriter::new(&file);

        stream.write_u32::<LittleEndian>(Self::MAGIC)?;
        stream.write_u32::<LittleEndian>(Self::VERSION)?;
        stream.write_u32::<LittleEndian>(self.hash_builder_seeds.len() as u32)?;
        for hash_builder_seed in &self.hash_builder_seeds {
            for seed in hash_builder_seed {
                stream.write_u64::<LittleEndian>(*seed)?;
            }
        }

        stream.write_u64::<LittleEndian>(self.bits.len() as u64)?;
        unsafe {
            let bytes: &[u8] = std::slice::from_raw_parts(
                self.bits.as_ptr() as *const u8,
                self.bits.len() * size_of::<AtomicU32>());
            stream.write_all(bytes)?;
        };

        Ok(())
    }

    pub fn hashes(&self, s: &VecDeque<&str>) -> Vec<u64> {
        self.hash_builders.iter().map(|hash_builder| {
            let mut hasher = hash_builder.build_hasher();
            s.hash(&mut hasher);
            hasher.finish()
        }).collect()
    }

    pub fn insert_hashes(&self, hashes: &Vec<u64>) {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            self.bits[index].fetch_or(1 << bit, Ordering::Relaxed);
        }
    }

    pub fn insert(&self, s: &VecDeque<&str>) {
        let hashes = self.hashes(s);
        self.insert_hashes(&hashes);
    }

    pub fn contains_hashes(&self, hashes: &Vec<u64>) -> bool {
        for hash in hashes {
            let hash = *hash as usize;
            let index = hash / 32 % self.bits.len();
            let bit = hash % 32;
            if self.bits[index].load(Ordering::Relaxed) & (1 << bit) == 0 {
                return false;
            }
        }

        return true;
    }

    pub fn contains(&self, s: &VecDeque<&str>) -> bool {
        let hashes = self.hashes(s);
        self.contains_hashes(&hashes)
    }

    pub fn initialize(config: &BloomFilterConfig) -> Result<BloomFilter, io::Error> {
        let save_file = PathBuf::from(&config.file);
        let bloom_filter = if save_file.exists() {
            log::info!("Loading bloom filter from {:?}...", save_file.display());
            BloomFilter::from_file(&save_file).unwrap()
        } else {
            log::info!("Creating new bloom filter...");
            let mut bloom_filter_size: usize = config.size_in_bytes;
            if bloom_filter_size <= 0 {
                bloom_filter_size = BloomFilter::suggest_size_in_bytes(config.estimated_doc_count, config.desired_false_positive_rate);
                log::info!("Creating bloom filter with size {} bytes to achieve false positive rate {} for {} elements", bloom_filter_size, config.desired_false_positive_rate, config.estimated_doc_count);
            }
            let num_hashers = BloomFilter::optimal_number_of_hashers(
                bloom_filter_size,
                config.estimated_doc_count);
            let p = BloomFilter::prob_of_false_positive(bloom_filter_size, config.estimated_doc_count, num_hashers);
            log::info!("Bloom filter will have size {}, {} hashers, false positive rate {}.", bloom_filter_size, num_hashers, p);
            BloomFilter::new(bloom_filter_size, num_hashers)
        };

        Ok(bloom_filter)
    }
}

