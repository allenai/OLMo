/* Copyright 2021 Google LLC
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


/* Create and use suffix arrays for deduplicating language model datasets.
 *
 * A suffix array A for a sequence S is a datastructure that contains all
 * suffixes of S in sorted order. To be space efficient, instead of storing
 * the actual suffix, we just store the pointer to the start of the suffix.
 * To be time efficient, it uses fancy algorithms to not require quadratic
 * (or worse) work. If we didn't care about either, then we could literally
 * just define (in python)
 * A = sorted(S[i:] for i in range(len(S)))
 *
 * Suffix arrays are amazing because they allow us to run lots of string
 * queries really quickly, while also only requiring an extra 8N bytes of
 * storage (one 64-bit pointer for each byte in the sequence).
 *
 * This code is designed to work with Big Data (TM) and most of the
 * complexity revolves around the fact that we do not require the
 * entire suffix array to fit in memory. In order to keep things managable,
 * we *do* require that the original string fits in memory. However, even
 * the largest language model datasets (e.g., C4) are a few hundred GB
 * which on todays machines does fit in memory.
 *
 * With all that amazing stuff out of the way, just a word of warning: this
 * is the first program I've ever written in rust. I still don't actually
 * understand what borrowing something means, but have found that if I
 * add enough &(&&x.copy()).clone() then usually the compiler just loses
 * all hope in humanity and lets me do what I want. I apologize in advance
 * to anyone actually does know rust and wants to lock me in a small room
 * with the Rust Book by Klabnik & Nichols until I repent for my sins.
 * (It's me, two months in the future. I now more or less understand how
 * to borrow. So now instead of the code just being all awful, you'll get
 * a nice mix of sane rust and then suddenly OH-NO-WHAT-HAVE-YOU-DONE-WHY!?!)
 */

use std::path::Path;
use std::time::Instant;
use std::fs;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use std::io::prelude::*;
use std::cmp::Reverse;
use std::convert::TryInto;

extern crate filebuffer;
extern crate zstd;
extern crate crossbeam;
extern crate clap;

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use clap::{Parser, Subcommand};

mod table;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {

    #[clap(arg_required_else_help = true)]
    Make {
        #[clap(short, long)]
        data_file: String,
    },

    MakePart {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        parts_dir: String,
        #[clap(short, long)]
        start_byte: usize,
        #[clap(short, long)]
        end_byte: usize,
    },

    CountOccurrences {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        query_file: String,
    },

    CountOccurrencesMulti {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        query_file: String,
    },

    SelfSimilar {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        length_threshold: usize,
        #[clap(short, long, default_value_t = 0)]
        frequency_threshold: usize,
        #[clap(short, long)]
        only_save_one: bool,
        #[clap(short, long)]
        cache_dir: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
    },

    AcrossSimilar {
        #[clap(long)]
        data_file_1: String,
        #[clap(long)]
        data_file_2: String,
        #[clap(short, long)]
        length_threshold: usize,
        #[clap(short, long)]
        cache_dir: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
    },

    Merge {
        #[clap(short, long)]
        suffix_path: Vec<String>,
        #[clap(short, long)]
        merged_dir: String,
        #[clap(short, long, default_value_t = 8)]
        num_threads: i64,
        #[clap(long, default_value_t = 100000)]
        hacksize: usize,
    },

    Collect {
        #[clap(short, long)]
        data_file: String,
        #[clap(short, long)]
        cache_dir: String,
        #[clap(short, long)]
        length_threshold: u64,
    }

}

/* Convert a uint64 array to a uint8 array.
 * This doubles the memory requirements of the program, but in practice
 * we only call this on datastructures that are smaller than our assumed
 * machine memory so it works.
 */
pub fn to_bytes(input: &[u64], size_width: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(size_width * input.len());

    for value in input {
        bytes.extend(&value.to_le_bytes()[..size_width]);
    }
    bytes
}

/* Convert a uint8 array to a uint64. Only called on (relatively) small files. */
pub fn from_bytes(input: Vec<u8>, size_width: usize) -> Vec<u64> {
    println!("S {}", input.len());
    assert!(input.len() % size_width == 0);
    let mut bytes:Vec<u64> = Vec::with_capacity(input.len()/size_width);

    let mut tmp = [0u8; 8];
    // todo learn rust macros, hope they're half as good as lisp marcos
    // and if they are then come back and optimize this
    for i in 0..input.len()/size_width {
        tmp[..size_width].copy_from_slice(&input[i*size_width..i*size_width+size_width]);
        bytes.push(u64::from_le_bytes(tmp));
    }

    bytes
}

/* For a suffix array, just compute A[i], but load off disk because A is biiiiiiigggggg. */
fn table_load_disk(table:&mut BufReader<File>,
                   index: usize,
                   size_width: usize) -> usize{
    table.seek(std::io::SeekFrom::Start ((index*size_width) as u64)).expect ("Seek failed!");
    let mut tmp = [0u8; 8];
    table.read_exact(&mut tmp[..size_width]).unwrap();
    return u64::from_le_bytes(tmp) as usize;
}

/* Binary search to find where query happens to exist in text */
// table: 4t bytes; text: 4t bytes; size_width = 4
fn off_disk_position(text: &[u8], table: &mut BufReader<File>,
                     query: &[u8], size_width: usize) -> usize {
    let dsize = std::mem::size_of::<u32>();

    let (mut left, mut right) = (0, text.len() / dsize);
    while left < right {
        let mid = (left + right) / 2;
        if query < &text[table_load_disk(table, mid, size_width)..] {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    left
}

/*
 * We're going to work with suffix arrays that are on disk, and we often want
 * to stream them top-to-bottom. This is a datastructure that helps us do that:
 * we read 1MB chunks of data at a time into the cache, and then fetch new data
 * when we reach the end.
 */
struct TableStream {
    file: BufReader<File>,
    cache: [u8; 8],
    size_width: usize
}

/* Make a table from a file path and a given offset into the table */
fn make_table(path: std::string::String,
              offset: usize,
              size_width: usize) -> TableStream {
    let mut table = TableStream {
        file: std::io::BufReader::with_capacity(1024*1024, fs::File::open(path).unwrap()),
        cache: [0u8; 8],
        size_width: size_width
    };
    table.file.seek (std::io::SeekFrom::Start ((offset*size_width) as u64)).expect ("Seek failed!");
    return table;
}

/* Get the next word from the suffix table. */
fn get_next_pointer_from_table_canfail(tablestream:&mut TableStream) -> u64 {
    let ok = tablestream.file.read_exact(&mut tablestream.cache[..tablestream.size_width]);
    let bad = match ok {
        Ok(_) => false,
        Err(_) => true,
    };
    if bad {
        return std::u64::MAX;
    }
    let out = u64::from_le_bytes(tablestream.cache);
    return out;
}


fn get_next_pointer_from_table(tablestream:&mut TableStream) -> u64 {
    let r = get_next_pointer_from_table_canfail(tablestream);
    if r == std::u64::MAX {
        panic!("Reached EOF badly");
    }
    return r;
}

fn table_load_filebuffer(table:&filebuffer::FileBuffer, index:usize, width: usize) -> usize{
    let mut tmp = [0u8; 8];
    tmp[..width].copy_from_slice(&table[index*width..index*width+width]);
    return u64::from_le_bytes(tmp) as usize;
}

/*
 * Helper function to actually do the count of the number of times something is repeated.
 * This should be fairly simple.
 * First, perform binary search using the on-disk suffix array to find the first place
 * where the string occurrs. If it doesn't exist then return 0.
 * Then, binary search again to find the last location it occurrs.
 * Return the difference between the two.
 */
fn count_occurances(text: &filebuffer::FileBuffer,
                    size_text: u64,
                    table: &filebuffer::FileBuffer,
                    size: u64,
                    str: &[u8],
                    size_width: usize,
                    print_where: bool) -> u64 {
    let mut buf: &[u8];
    assert!(size % (size_width as u64) == 0);

    let mut low = 0;
    let mut high = size/(size_width as u64);
    while low < high {
        let mid = (high+low)/2;
        let pos = table_load_filebuffer(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos+str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str <= &buf {
            high = mid;
        } else {
            low = mid+1;
        }
    }
    let start = low;

    let pos = table_load_filebuffer(&table, low as usize, size_width);
    if pos + str.len() < size_text as usize {
        buf = &text[pos..pos+str.len()];
    } else {
        buf = &text[pos..size_text as usize];
    }

    if str != buf {
        return 0; // not found
    }

    if print_where {
        println!("Found at: {}", pos);
    }

    high = size/(size_width as u64);
    while low < high {
        let mid = (high+low)/2;
        let pos = table_load_filebuffer(&table, mid as usize, size_width);

        if pos + str.len() < size_text as usize {
            buf = &text[pos..pos+str.len()];
        } else {
            buf = &text[pos..size_text as usize];
        }

        if str != buf {
            high = mid;
        } else {
            low = mid+1;
        }
    }
    return low-start;
}

/*
 * Create a suffix array for a given file in one go.
 * Calling this method is memory heavy---it's technically linear in the
 * length of the file, but the constant is quite big.
 * As a result, this method should only be called for files that comfortably
 * fit into memory.
 *
 * The result of calling this method is a new file with ".table.bin" appended
 * to the name which is the suffix array of sorted suffix pointers. This file
 * should be at most 8x larger than the original file (one u64 pointer per
 * byte of the original). In order to save space, if it turns out we only need
 * 32 bits to uniquely point into the data file then we serialize using fewer
 * bits (or 24, or 40, or ...), but in memory we always use a u64.
 *
 * If the file does not fit into memory, then instead you should use the
 * alternate save_part and then merge_parallel in two steps. See the comments
 * below for how those work.
 */
fn cmd_make(fpath: &String)   -> std::io::Result<()> {
    let dsize = std::mem::size_of::<u32>();

    let now = Instant::now();
    println!("Reading the dataset at time t={}ms", now.elapsed().as_millis());
    let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
    fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_)?;

    // LJC: Cast the buffers to u32 so we can build the SA for valid positions only
    assert!(text_.len() % dsize == 0);
    // LJC: We have to use big-endian here to interpret into u32, because the suffix array operates on bytes
    let u32_text_: Vec<u32> = text_
        .chunks(dsize)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let u32_text = &u32_text_;

    let text = &text_;
    println!("Done reading the dataset at time t={}ms", now.elapsed().as_millis());

    println!("... and now starting the suffix array construction.");


    let st = table::SuffixTable::new(u32_text);
    println!("Done building suffix array at t={}ms",now.elapsed().as_millis());
    let parts = st.into_parts();
    let table = parts.1;

    // LJC: multiply every element of the table by dsize, because the offsets are counted in bytes
    let table2 = table.iter().map(|x| x * dsize as u64).collect::<Vec<u64>>();

    let ratio = ((text.len() as f64).log2()/8.0).ceil() as usize;
    println!("Ratio: {}", ratio);

    let mut buffer = File::create(fpath.clone() + ".table.bin")?;
    let bufout = to_bytes(&table2, ratio);
    println!("Writing the suffix array at time t={}ms", now.elapsed().as_millis());
    buffer.write_all(&bufout)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

/*
 * Create a suffix array for a subsequence of bytes.
 * As with save, this method is linear in the number of bytes that are
 * being saved but the constant is rather high. This method does exactly
 * the same thing as save except on a range of bytes.
 */
fn cmd_make_part(fpath: &String, parts_dir: &String, start: u64, end: u64)   -> std::io::Result<()> {
    let dsize = std::mem::size_of::<u32>();

    let now = Instant::now();
    println!("Opening up the dataset files");

    let space_available = std::fs::metadata(fpath.clone()).unwrap().len() as u64;
    assert!(start < end);
    assert!(end <= space_available);

    let mut text_ = vec![0u8; (end-start) as usize];
    let mut file = fs::File::open(fpath.clone()).unwrap();
    println!("Loading part of file from byte {} to {}", start, end);
    file.seek(std::io::SeekFrom::Start(start)).expect ("Seek failed!");
    file.read_exact(&mut text_).unwrap();

    // LJC: Cast the buffers to u32 so we can build the SA for valid positions only
    assert!(text_.len() % dsize == 0);
    // LJC: We have to use big-endian here to interpret into u32, because the suffix array operates on bytes
    let u32_text_: Vec<u32> = text_
        .chunks(dsize)
        .map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let u32_text = &u32_text_;

    let text = &text_;
    println!("Done reading the dataset at time t={}ms", now.elapsed().as_millis());
    println!("... and now starting the suffix array construction.");

    let st = table::SuffixTable::new(u32_text);
    println!("Done building suffix array at t={}ms",now.elapsed().as_millis());
    let parts = st.into_parts();
    let table = parts.1;

    // LJC: multiply every element of the table by dsize, because the offsets are counted in bytes
    let table2 = table.iter().map(|x| x * dsize as u64).collect::<Vec<u64>>();

    let ratio = ((text.len() as f64).log2()/8.0).ceil() as usize;
    println!("Ratio: {}", ratio);

    let mut buffer = File::create(format!("{}/{}-{}.table.bin", parts_dir, start, end))?;
    let mut buffer2 = File::create(format!("{}/{}-{}", parts_dir, start, end))?;
    let bufout = to_bytes(&table2, ratio);
    println!("Writing the suffix array at time t={}ms", now.elapsed().as_millis());
    buffer.write_all(&bufout)?;
    buffer2.write_all(text)?;
    println!("And finished at time t={}ms", now.elapsed().as_millis());
    Ok(())
}

/*
 * Count how many times a particular string has occurred in the dataset.
 *
 * This is the easiest method to understand. It just performs binary search on the
 * suffix array and uses it exactly as it was designed. It will output the number of counts.
 *
 * NOTE: This function allows overlapping sequences to count as different duplicates.
 * So if our string is `aaaa` and we count how many times `aa` occurrs, it will return 3,
 * not 2. This is different from python's "aaaa".count("aa") which will say 2.
 * This may or may not be a problem for you. But if is is, that's you're problem, not mine.
 */
fn cmd_count_occurrences(fpath: &String, querypath: &String)   -> std::io::Result<()> {
    /* Count the numberof times a particular sequence occurs in the table.
     */

    let metadata_text = fs::metadata(format!("{}", fpath))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", fpath))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    let text = filebuffer::FileBuffer::open(fpath).unwrap();
    let table = filebuffer::FileBuffer::open(format!("{}.table.bin", fpath)).unwrap();

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let mut str = Vec::with_capacity(std::fs::metadata(querypath.clone()).unwrap().len() as usize);
    fs::File::open(querypath.clone()).unwrap().read_to_end(&mut str)?;

    let occurances = count_occurances(&text, size_text,  &table, size_table, &str[0..str.len()], size_width as usize, false);

    println!("Number of times present: {}\n", occurances);
    Ok(())
}

/*
 * Count the number of times a particular sequence occurs in the table.
 * (for multiple queries)
 */
fn cmd_count_occurrences_multi(fpath: &String, querypath: &String)   -> std::io::Result<()> {

    let metadata_text = fs::metadata(format!("{}", fpath))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", fpath))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    let text = filebuffer::FileBuffer::open(fpath).unwrap();
    let table = filebuffer::FileBuffer::open(format!("{}.table.bin", fpath)).unwrap();

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let mut str = Vec::with_capacity(std::fs::metadata(querypath.clone()).unwrap().len() as usize);
    fs::File::open(querypath.clone()).unwrap().read_to_end(&mut str)?;

    let mut off = 0;
    while off < str.len() {
        let length = u32::from_le_bytes(str[off..off+4].try_into().expect("?")) as usize;
        off += 4;

        let occurances = count_occurances(&text, size_text, &table, size_table, &str[off..off+length], size_width as usize, true);
        off += length;
        println!("Number of times present: {}", occurances);
    }
    Ok(())
}


/*
 * Given a string S and suffix array A, compute statistics about how many
 * sequences in A are duplicated (and do it using as many threads as possible).
 *
 * The basic algorithm is simple. For every pair of items (i,i+1) in the
 * suffix array, we compare the suffixes S[A[i]..] and S[A[i+i]..] and count
 * how many characters they have in common. We then report various statistics
 * about this (e.g., the length of the match, which sequences match each other
 * with at least T tokens, etc).
 *
 * The first complication is that we can't load all of A into memory at once.
 * This is too big. (e.g., the suffix array for C4 is 2.7 terabytes (!).
 * We might be able to fit 345GB in memory on current hardware, but not
 * 2.7TB. (If you're reading this in 2030, hello there. This must all look
 * very silly to you. But I promise that, today, 2.7TB of memory is just too
 * much. By the way, has AGI taken over the world? I hope not.)
 *
 * Fortunately our algorithm doesn't require random access into A, so we can
 * just stream it off disk and then immediately throw away the old data.
 *
 * The second complication is that we want this to be fast. Very fast. So
 * we're going to parallelize the algorithm over as many threads as possible.
 * Fortunately this is Rust, and not Python, so the GIL is not going to make
 * life terrible. We set up one copy of the string S in memory, and then we
 * can have each of the threads in parallel stream over A starting at different
 * offsets.
 *
 * The output of this algorithm is a bunch of files saved to cache_dir named
 * /cache_dir/dups_S_i-j
 * /cache_dir/sizes_S_i-j
 * Where i-j is the range of bytes that are covered by this file.
 * The dups file stores just a list of 8-byte values [x_i] of indexs where S[x..x+T]
 * is duplicated elsewhere in the dataset.
 *
 * Because the list is produced in lexical order, the duplicates for the same string
 * will all be sequential in the list, and this is where the sizes file comes in.
 * The sizes file says which duplicates from the dups file correspond to the same "cluster".
 * So if sizes = [5, 2, 8 ...] then it means the first 5 entries in the dups file correspond
 * to the same string that's repeated 5 times, and the next 2 entries in the dups file are
 * a pair of repeated strings.
 */
fn cmd_self_similar(data_file: &String, length_threshold: &usize, frequency_threshold: &usize,
                    only_save_one: &bool, cache_dir: &String, num_threads: i64)  -> std::io::Result<()> {
    println!("Start load!");

    let text = filebuffer::FileBuffer::open(data_file).unwrap();

    let metadata = fs::metadata(format!("{}.table.bin", data_file))?;

    assert!(metadata.len() % (text.len() as u64) == 0);

    let ratio = metadata.len()/(text.len() as u64);

    if !Path::new(&cache_dir).exists() {
        fs::create_dir(cache_dir)?;
    }

    fn worker(text:&[u8], start:usize, end:usize,
              length_threshold: usize, frequency_threshold: usize, only_save_one: bool,
              data_file: String, cache_dir: String,
              ratio: usize) -> usize {
        let mut table = make_table(format!("{}.table.bin", data_file), start, ratio);
        let mut prev_location = get_next_pointer_from_table(&mut table);

        let mut outfile = std::io::BufWriter::new(fs::File::create(
            format!("{}/dups_{}_{}-{}", cache_dir,
                    data_file.split("/").last().unwrap(), start, end)).unwrap());
        let mut outfile_sizes = std::io::BufWriter::new(fs::File::create(
            format!("{}/sizes_{}_{}-{}", cache_dir,
                    data_file.split("/").last().unwrap(), start, end)).unwrap());

        let mut duplicate_count = 0;
        let mut i = start;
        let mut pairs:Vec<u64> = Vec::with_capacity(4);

        while i < end {
            if i%1000000000 == 0 { println!("{} / {} ", i-start, end-start); }
            let suf1 = &text[prev_location as usize..];

            let mut cur_location;

            let mut first = true;

            loop {
                cur_location = get_next_pointer_from_table_canfail(&mut table);
                i += 1;
                if cur_location == std::u64::MAX {
                    // The last two items in the file matched
                    break;
                }

                let suf2 = &text[cur_location as usize..];
                let does_match =  suf2.len() >= length_threshold && suf1.len() >= length_threshold && suf1[..length_threshold] == suf2[..length_threshold];
                if does_match {
                    if !first {
                        pairs.push(cur_location);
                    } else {
                        pairs.push(prev_location);
                        pairs.push(cur_location);
                        first = false;
                    }
                } else {
                    break;
                }
            }

            if pairs.len() > frequency_threshold {
                if only_save_one {
                    let seq = &text[pairs[0] as usize..pairs[0] as usize+length_threshold];
                    if pairs[0]%2 == 0 {
                        outfile.write_all(seq).expect("Ok");
                    }
                } else {
                    outfile.write_all(&to_bytes(&pairs[..], ratio)[..]).expect("Ok");
                    outfile_sizes.write_all(&to_bytes(&[pairs.len() as u64][..], ratio)[..]).expect("Ok");
                    duplicate_count += pairs.len();
                }
            }
            pairs.clear();

            prev_location = cur_location;
        }

        return duplicate_count;
    }

    let now = Instant::now();

    let increment:i64 = (text.len() as i64-num_threads)/num_threads;
    let _answer = crossbeam::scope(|scope| {
        let mut result = Vec::with_capacity(num_threads as usize);
        let text = &text;
        for i in 0..num_threads {
            let one_result = scope.spawn(move || {
                return worker(text,
                              std::cmp::max(0i64,i*increment-1) as usize,
                              std::cmp::min(((i+1)*increment) as usize, text.len()),
                              *length_threshold, *frequency_threshold, *only_save_one,
                              data_file.clone(), cache_dir.clone(),
                              ratio as usize);
            });
            result.push(one_result);
        }

        let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
        println!("Duplicates found: {:?}", thread_sum);

    });

    println!("Total time taken: {}ms", now.elapsed().as_millis());

    Ok(())
}

/*
 * Given a string S1 and suffix array A1, and another string S2 with array A2,
 * find all sequences that are duplicated between S1 and S2 with any particular length.
 *
 * The basic algorithm is simple, and seems very much like a merge operation.
 * Start enumerating all sequences from A1 which gives a sorted enumeration of S1.
 * If S1[A1[0]..] < S2[A2[0]..] then advance the pointer walking S1, otherwise
 * advance the pointer walking S2. If ever S1[A1[i]..A[i]+L] = S2[A2[j]..A2[j]+L]
 * then we have a match and write it down.
 *
 * As with the self-similar comparison, we can't fit A1 or A2 into memory. So do the
 * same streming tricks. And again we want things to go fast, so we're going to run
 * it on as many parallel threads as possible.
 *
 * The output of this algorithm is a bunch of files saved to cache_dir named
 * /cache_dir/dups_S1_i-j_S1-k-l
 * /cache_dir/sizes_S2_i-j_S2-k-l
 * Here, A and B are the two files we're cross-deduplicating (probably a train and test set).
 * i-j is the range of bytes that are covered by this file in S1, and similarly k-l for S2.
 *
 * The dups and size file have the same interpretation as before. But this time there are
 * two, one for the A -> B comparison, and another for the B -> A comparison.
 */
fn cmd_across_similar(data_file_1: &String, data_file_2: &String, cache_dir: &String,
                      length_threshold: usize, num_threads: i64)  -> std::io::Result<()> {
    let text1 = filebuffer::FileBuffer::open(data_file_1).unwrap();
    let text2 = filebuffer::FileBuffer::open(data_file_2).unwrap();

    let metadata1 = fs::metadata(format!("{}.table.bin", data_file_1)).expect("suffix array exists for arg 0");
    let metadata2 = fs::metadata(format!("{}.table.bin", data_file_2)).expect("suffix array exists for arg 1");

    assert!(metadata1.len() % (text1.len() as u64) == 0);
    let ratio1 = metadata1.len()/(text1.len() as u64);

    assert!(metadata2.len() % (text2.len() as u64) == 0);
    let ratio2 = metadata2.len()/(text2.len() as u64);

    if !Path::new(&cache_dir).exists() {
        fs::create_dir(cache_dir)?;
    }

    fn worker(text1:&[u8], text2:&[u8],
              start1:usize, end1:usize,
              start2:usize, end2:usize,
              data_file_1: String, data_file_2: String,
              cache_dir: String, length_threshold: usize,
              size_width_1: usize, size_width_2: usize) -> usize {
        let mut table1 = make_table(format!("{}.table.bin", data_file_1), start1, size_width_1);
        let mut location1 = get_next_pointer_from_table(&mut table1);

        let mut table2 = make_table(format!("{}.table.bin", data_file_2), start2, size_width_2);
        let mut location2 = get_next_pointer_from_table(&mut table2);

        // What do you mean this looks ugly. I see no problem here!
        let mut outfile1 = std::io::BufWriter::new(fs::File::create(
            format!("{}/dups_{}_{}-{}_{}_{}-{}",
                    cache_dir,
                    data_file_1.split("/").last().unwrap(), start1, end1,
                    data_file_2.split("/").last().unwrap(), start2, end2,
            )).unwrap());
        let mut outfile1_sizes = std::io::BufWriter::new(fs::File::create(
            format!("{}/sizes_{}_{}-{}_{}_{}-{}",
                    cache_dir,
                    data_file_1.split("/").last().unwrap(), start1, end1,
                    data_file_2.split("/").last().unwrap(), start2, end2,
            )).unwrap());

        let mut outfile2 = std::io::BufWriter::new(fs::File::create(
            format!("{}/dups_{}_{}-{}_{}_{}-{}",
                    cache_dir,
                    data_file_2.split("/").last().unwrap(), start2, end2,
                    data_file_1.split("/").last().unwrap(), start1, end1,
            )).unwrap());
        let mut outfile2_sizes = std::io::BufWriter::new(fs::File::create(
            format!("{}/sizes_{}_{}-{}_{}_{}-{}",
                    cache_dir,
                    data_file_2.split("/").last().unwrap(), start2, end2,
                    data_file_1.split("/").last().unwrap(), start1, end1,
            )).unwrap());


        let mut duplicate_count = 0;
        let mut i = start1;
        let mut j = start2;
        while i < end1 && j < end2 {
            if (i+j)%1000000000 == 0 { println!("{} / {} ", i, text1.len()); }

            let mut suf1 = &text1[location1 as usize..];
            let mut suf2 = &text2[location2 as usize..];

            // Do we have a match between the suffix that begins at location1 in text1
            // and the suffix that begins at location2 in text2?
            // To check this we need (a) both are long enough, and
            // (b) the match is of length at least length_threshold

            let does_match = suf1.len() >= length_threshold && suf2.len() >= length_threshold && suf1[..length_threshold] == suf2[..length_threshold];

            if does_match {
                // We have a match between a subsequence in text1 and text2
                let target_suf = &suf1[..length_threshold]; // wlog. equals suf2[..length_threshold]

                // We want the matches to be clustered, so let's find all matches from
                // the first string that are equal to target_suf
                let start = i;
                while suf1.len() >= length_threshold && &suf1[..length_threshold] == target_suf {
                    outfile1.write_all(&to_bytes(&[location1 as u64][..], size_width_1)[..]).expect("Ok");

                    location1 = get_next_pointer_from_table_canfail(&mut table1);
                    i += 1;
                    if location1 == std::u64::MAX {
                        break;
                    }
                    suf1 = &text1[location1 as usize..];
                }
                duplicate_count += i-start;
                outfile1_sizes.write_all(&to_bytes(&[(i-start) as u64][..], size_width_1)[..]).expect("Ok");

                // And now find all matches from the second string that are equal to target_suf
                let start = j;
                while suf2.len() >= length_threshold && &suf2[..length_threshold] == target_suf {
                    outfile2.write_all(&to_bytes(&[location2 as u64][..], size_width_2)[..]).expect("Ok");

                    location2 = get_next_pointer_from_table(&mut table2);
                    j += 1;
                    if location2 == std::u64::MAX {
                        break;
                    }
                    suf2 = &text2[location2 as usize..];
                }
                duplicate_count += j-start;
                outfile2_sizes.write_all(&to_bytes(&[(j-start) as u64][..], size_width_2)[..]).expect("Ok");
            } else if suf1 < suf2 {
                // No match, and the first suffix is smaller. Increment the smaller one
                i += 1;
                location1 = get_next_pointer_from_table_canfail(&mut table1);
            } else if suf2 < suf1 {
                // No match, and the second suffix is smaller. Increment the smaller one
                j += 1;
                location2 = get_next_pointer_from_table_canfail(&mut table2);
            } else {
                // This happens only when
                // 1. The two suffixes are identical
                // 2. But they're not yet long enough for it to "count"
                // so we just increment one of the poitners WLOG
                assert!(&suf1 == &suf2);
                assert!(suf1.len() < 100 || suf2.len() < 100);
                i += 1;
                location1 = get_next_pointer_from_table(&mut table1);
            }
        }

        return duplicate_count;
    }


    // Start a bunch of jobs that each work on non-overlapping regions of the suffix array.
    let increment:i64 = (text1.len() as i64-num_threads)/num_threads;
    let _answer = crossbeam::scope(|scope| {
        let mut result = Vec::with_capacity(num_threads as usize);
        let text1 = &text1;
        let text2 = &text2;
        let mut last_end = 0;
        for i in 0..num_threads {
            let a = std::cmp::max(0i64,i*increment-1) as usize;
            let b = std::cmp::min(((i+1)*increment) as usize, text1.len());

            let mut table1 = std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_file_1)).unwrap());
            let mut table2 = std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_file_2)).unwrap());
            let this_start = last_end;

            let end_seq = &text1[table_load_disk(&mut table1, b, ratio1 as usize)..];
            let this_end = off_disk_position(text2, &mut table2, end_seq, ratio2 as usize);

            last_end = this_end;
            println!("start {} {}", this_start, this_end);
            let one_result = scope.spawn(move || {

                return worker(text1, text2,
                              a, b,
                              this_start, this_end,
                              data_file_1.clone(), data_file_2.clone(),
                              cache_dir.clone(),
                              length_threshold,
                              ratio1 as usize, ratio2 as usize);
            });
            result.push(one_result);
        }

        let thread_sum:usize = result.into_iter().map(|t| t.join()).sum();
        println!("Duplicates found: {:?}", thread_sum);

    });
    Ok(())
}


/*
 * A little bit of state for the merge operation below.
 * - suffix is suffix of one of the parts of the dataset we're merging;
this is the value we're sorting on
 * - position is the location of this suffix (so suffix = array[position..])
 * - table_index says which suffix array this suffix is a part of
 */
#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeState<'a> {
    suffix: &'a [u8],
    position: u64,
    table_index: usize,
    hacksize: usize,
}

impl<'a> Ord for MergeState<'a> {
    fn cmp(&self, other: &Self) -> Ordering {
        other.suffix[..other.suffix.len().min(other.hacksize)].cmp(&self.suffix[..self.suffix.len().min(self.hacksize)])
    }
}

impl<'a> PartialOrd for MergeState<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}


/*
 * Merge together M different suffix arrays (probably created with make-part).
 * That is, given strings S_i and suffix arrays A_i compute the suffix array
 * A* = make-suffix-array(concat S_i)
 * In order to do this we just implement mergesort's Merge operation on each
 * of the arrays A_i to construct a sorted array A*.
 *
 * This algorithm is *NOT A LINEAR TIME ALGORITHM* in the worst case. If you run
 * it on a dataset consisting entirely of the character A it will be quadratic.
 * Fortunately for us, language model datasets typically don't just repeat the same
 * character a hundred million times in a row. So in practice, it's linear time.
 *
 * There are thre complications here.
 *
 * As with selfsimilar_parallel, we can't fit all A_i into memory at once, and
 * we want to make things fast and so parallelize our execution. So we do the
 * same tricks as before to make things work.
 *
 * However we have one more problem. In order to know how to merge the final
 * few bytes of array S_0 into their correct, we need to know what bytes come next.
 * So in practice we make sure that S_{i}[-HACKSIZE:] === S_{i+1}[:HACKSIZE].
 * As long as HACKSIZE is longer than the longest potential match, everything
 * will work out correctly. (I did call it hacksize after all.....)
 * In practice this works. It may not for your use case if there are long duplicates.
 */
fn cmd_merge(data_files: &Vec<String>, merged_dir: &String, num_threads: i64, hacksize: usize)  -> std::io::Result<()> {
    // This value is declared here, but also in scripts/make_suffix_array.py
    // If you want to change it, it needs to be changed in both places.

    let dsize = std::mem::size_of::<u32>();

    let nn:usize = data_files.len();

    fn load_text2<'s,'t>(fpath:String) -> Vec<u8> {
        println!("Setup buffer");
        let mut text_ = Vec::with_capacity(std::fs::metadata(fpath.clone()).unwrap().len() as usize);
        println!("Done buffer {}", text_.len());
        fs::File::open(fpath.clone()).unwrap().read_to_end(&mut text_).unwrap();
        println!("Done read buffer");
        return text_;
    }

    // Start out by loading the data files and suffix arrays.
    let texts:Vec<Vec<u8>> = (0..nn).map(|x| load_text2(data_files[x].clone())).collect();

    let texts_len:Vec<usize> = texts.iter().enumerate().map(|(i,x)| x.len() - (if i+1 == texts.len() {0} else {hacksize})).collect();

    let metadatas:Vec<u64> = (0..nn).map(|x| {
        let meta = fs::metadata(format!("{}.table.bin", data_files[x].clone())).unwrap();
        assert!(meta.len()%(texts[x].len() as u64 / dsize as u64) == 0);
        return meta.len();
    }).collect();

    let big_ratio = ((texts_len.iter().sum::<usize>() as f64).log2()/8.0).ceil() as usize; // size of merged pointer in bytes
    println!("Ratio: {}", big_ratio);

    let ratio = metadatas[0] / (texts[0].len() as u64 / dsize as u64); // size of part pointer in bytes

    fn worker(texts:&Vec<Vec<u8>>, starts:Vec<usize>, ends:Vec<usize>, texts_len:Vec<usize>, part:usize,
              merged_dir: String, data_files: Vec<String>, ratio: usize, big_ratio: usize, hacksize: usize) {
        // starts and ends is counting by token

        let nn = texts.len();
        let mut tables:Vec<TableStream> = (0..nn).map(|x| {
            make_table(format!("{}.table.bin", data_files[x]), starts[x], ratio)
        }).collect();

        let mut idxs:Vec<u64> = starts.iter().map(|&x| x as u64).collect();

        let delta:Vec<u64> = (0..nn).map(|x| {
            let pref:Vec<u64> = texts[..x].iter().map(|y| y.len() as u64).collect();
            pref.iter().sum::<u64>() - (hacksize * x) as u64
        }).collect();

        let mut next_table = std::io::BufWriter::new(File::create(format!("{}/{:04}", merged_dir.clone(), part)).unwrap());

        fn get_next_maybe_skip(mut tablestream:&mut TableStream,
                               index:&mut u64, thresh:usize) -> u64 {
            //println!("{}", *index);
            let mut location = get_next_pointer_from_table_canfail(&mut tablestream);
            if location == u64::MAX {
                return location;
            }
            *index += 1;
            while location >= thresh as u64 {
                location = get_next_pointer_from_table_canfail(&mut tablestream);
                if location == u64::MAX {
                    return location;
                }
                *index += 1;
            }
            return location;
        }

        let mut heap = BinaryHeap::new();

        for x in 0..nn {
            let position = get_next_maybe_skip(&mut tables[x],
                                               &mut idxs[x], texts_len[x]);
            //println!("{} @ {}", position, x);
            if idxs[x] <= ends[x] as u64 {
                heap.push(MergeState {
                    suffix: &texts[x][position as usize..],
                    position: position,
                    table_index: x,
                    hacksize: hacksize
                });
            }
        }

        // Our algorithm is not linear time if there are really long duplicates
        // found in the merge process. If this happens we'll warn once.
        let mut did_warn_long_sequences = false;

        let mut prev = &texts[0][0..];
        while let Some(MergeState {suffix: _suffix, position, table_index, hacksize}) = heap.pop() {
            //next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()).expect("Write OK");
            next_table.write_all(&(position + delta[table_index] as u64).to_le_bytes()[..big_ratio]).expect("Write OK");

            let position = get_next_maybe_skip(&mut tables[table_index],
                                               &mut idxs[table_index], texts_len[table_index],);
            if position == u64::MAX {
                continue;
            }

            if idxs[table_index] <= ends[table_index] as u64 {
                let next = &texts[table_index][position as usize..];
                //println!("  {:?}", &next[..std::cmp::min(10, next.len())]);

                let match_len = (0..(hacksize+1)).find(|&j| !(j < next.len() && j < prev.len() && next[j] == prev[j]));
                if !did_warn_long_sequences {
                    if let Some(match_len_) = match_len {
                        if match_len_ >= hacksize {
                            println!("There is a match longer than 50,000,000 bytes.");
                            println!("You probably don't want to be using this code on this dataset---it's (possibly) quadratic runtime now.");
                            did_warn_long_sequences = true;
                        }
                    } else {
                        println!("There is a match longer than 50,000,000 bytes.");
                        println!("You probably don't want to be using this code on this dataset---it's quadratic runtime now.");
                        did_warn_long_sequences = true;
                    }
                }

                heap.push(MergeState {
                    suffix: &texts[table_index][position as usize..],
                    position: position,
                    table_index: table_index,
                    hacksize: hacksize
                });
                prev = next;
            }
        }
    }

    // Make sure we have enough space to take strided offsets for multiple threads
    // This should be an over-approximation, and starts allowing new threads at 1k of data
    let num_threads = std::cmp::min(num_threads, std::cmp::max((texts[0].len() as i64 - 1024)/10, 1));
    println!("AA {}", num_threads);

    // Start a bunch of jobs that each work on non-overlapping regions of the final resulting suffix array
    // Each job is going to look at all of the partial suffix arrays to take the relavent slice.
    let _answer = crossbeam::scope(|scope| {

        let mut tables:Vec<BufReader<File>> = (0..nn).map(|x| {
            std::io::BufReader::new(fs::File::open(format!("{}.table.bin", data_files[x])).unwrap())
        }).collect();

        let mut starts = vec![0; nn];

        for i in 0..num_threads as usize {
            let texts = &texts;
            let mut ends: Vec<usize> = vec![0; nn];
            if i < num_threads as usize-1 {
                ends[0] = (texts[0].len()/dsize+(num_threads as usize))/(num_threads as usize)*(i+1);
                let end_seq = &texts[0][table_load_disk(&mut tables[0], ends[0], ratio as usize)..];

                for j in 1..ends.len() {
                    ends[j] = off_disk_position(&texts[j], &mut tables[j], end_seq, ratio as usize);
                }
            } else {
                for j in 0..ends.len() {
                    ends[j] = texts[j].len()/dsize;
                }
            }

            for j in 0..ends.len() {
                let l = &texts[j][table_load_disk(&mut tables[j], starts[j], ratio as usize)..];
                let l = &l[..std::cmp::min(l.len(), 20)];
                println!("Text{} {:?}", j, l);
            }

            // let mut total = 0;
            // for j in 0..ends.len() {
            //     total += ends[j] - starts[j];
            // }
            // println!("Spawn {}: {}", i, total);

            println!("Spawn {}: {:?} {:?}", i, starts, ends);

            // if i == 52 as usize {
            let starts2 = starts.clone();
            let ends2 = ends.clone();
            //println!("OK {} {}", starts2, ends2);
            let texts_len2 = texts_len.clone();
            let _one_result = scope.spawn(move || {
                worker(texts,
                       starts2,
                       ends2,
                       texts_len2,
                       i,
                       (*merged_dir).clone(),
                       (*data_files).clone(),
                       ratio as usize,
                       big_ratio as usize,
                       hacksize
                );
            });
            // }

            for j in 0..ends.len() {
                starts[j] = ends[j];
            }
        }
    });

    // LJC: commenting these out as it appears to be a redundant file write
    // println!("Finish writing");
    // let mut buffer = File::create(output_file)?;
    // for i in 0..texts.len()-1 {
    //     buffer.write_all(&texts[i][..texts[i].len()-HACKSIZE])?;
    // }
    // buffer.write_all(&texts[texts.len()-1])?;
    Ok(())
}

/*
 * Given the output of either self-similar or across-similar,
 * compute byte ranges that are duplicates.
 *
 * The similar outputs are just byte values
 * [A_0, A_1, ..., A_N]
 * meaning that the bytes from (A_i, A_i + length_threshold) are duplicated somewhere.
 *
 * This script converts this to ranges [a, b) for complete ranges that should be removed.
 * For example if we have a long duplicate sequence
 *    abcdefg
 * then we might have a match for `abcde` and `bcdef` and `cdefg`
 * So instead of just saying tokens 0, 1, and 2 match, here we say that [0, 7) match.
 *
 * To do this we
 *   (a) sort the output lists, and then
 *   (b) collapse overlapping buckets.
 *
 * Note that as a result of doing this, we might have a sequence `qwerty` where the
 * entire sequence is never repeated in the dataset multiple times, but each byte
 * in the sequence is part of some length_threshold duplicate.
 */
fn cmd_collect(data_file: &String, cache_dir: &String, length_threshold: u64)  -> std::io::Result<()> {
    let paths = fs::read_dir(cache_dir).unwrap();

    let metadata_text = fs::metadata(format!("{}", data_file))?;
    let metadata_table = fs::metadata(format!("{}.table.bin", data_file))?;
    let size_text = metadata_text.len();
    let size_table = metadata_table.len();

    assert!(size_table % size_text == 0);
    let size_width = size_table / size_text;

    let ds_name = data_file.split("/").last().unwrap();


    let mut path_list = Vec::with_capacity(1000);
    for path in paths {
        let path = path.unwrap().path().as_path().to_str().unwrap().to_string();
        if !path.starts_with(&Path::new(cache_dir).join(format!("dups_{}_", ds_name)).into_os_string().into_string().unwrap()) {
            continue;
        }
        path_list.push(path);
    }

    // 1. Perform an initial sort of each of the found duplicates

    let mut result = Vec::with_capacity(100);
    crossbeam::scope(|scope| {
        for path in path_list.into_iter() {
            let path = path.clone();
            let out = scope.spawn(move || {
                let mut all_items = from_bytes(fs::read(path.clone()).unwrap(), size_width as usize);
                //println!("Got {} {:?}", size_width, &all_items[..10]);

                //let mut all_items:Vec<u64> = all_items.into_iter().filter(|&x| x%2 == 0).collect();
                all_items.sort_unstable();
                //println!("Done {}", all_items.len());
                return all_items;
            });
            result.push(out);
        }
    });
    let outputs:Vec<Vec<u64>> = result.into_iter().map(|t| t.join()).collect();

    let mut all_items:Vec<u64> = Vec::with_capacity(1000);
    println!("Merging.");

    // 2. Perform a merge of the now-sorted lists

    let mut heap = BinaryHeap::new();

    // Seed the heap with the first element of each
    for (i, output) in outputs.iter().enumerate() {
        if output.len() > 0 {
            heap.push(Reverse((output[0], 0, i)));
        }
    }

    let mut ranges:Vec<(u64,u64)> = Vec::with_capacity(1000);
    let mut prev_start;
    let mut prev_end;

    // Unroll first iteration of the loop for performance
    if let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
        prev_start = data_pointer;
        prev_end = data_pointer + length_threshold;
	// ensure this bucket has enough data to push the item
        if index+1 < outputs[which_array].len() {
            heap.push(Reverse((outputs[which_array][index+1], index+1, which_array)));
	}
    } else {
        println!("No duplicates found! Either the dataset is duplicate-free or something went wrong.");
        return Ok(());
    }

    // Now walk the the rest of the merging
    while let Some(Reverse((data_pointer, index, which_array))) = heap.pop() {
        all_items.push(data_pointer);

        if data_pointer <= prev_end {
            prev_end = data_pointer+length_threshold;
        } else {
            ranges.push((prev_start, prev_end));
            prev_start = data_pointer;
            prev_end = data_pointer+length_threshold;
        }

        // If this array has more data, consume it
        if index+1 < outputs[which_array].len() {
            heap.push(Reverse((outputs[which_array][index+1], index+1, which_array)));
        }
    }
    ranges.push((prev_start, prev_end));

    let strout:Vec<String> = ranges.iter().map(|&x| format!("{} {}", x.0, x.1)).collect();
    println!("out\n{}", strout.join("\n"));
    Ok(())
}

fn main()  -> std::io::Result<()> {

    let args = Args::parse();


    match &args.command {
        Commands::Make { data_file } => {
            cmd_make(data_file)?;
        }

        Commands::MakePart { data_file, parts_dir, start_byte, end_byte } => {
            cmd_make_part(data_file, parts_dir, *start_byte as u64, *end_byte as u64)?;
        }

        Commands::CountOccurrences { data_file, query_file } => {
            cmd_count_occurrences(data_file,
                                  query_file)?;
        }

        Commands::CountOccurrencesMulti { data_file, query_file } => {
            cmd_count_occurrences_multi(data_file,
                                        query_file)?;
        }

        Commands::SelfSimilar { data_file, length_threshold, frequency_threshold, only_save_one, cache_dir, num_threads } => {
            cmd_self_similar(data_file, length_threshold, frequency_threshold, only_save_one, cache_dir, *num_threads)?;
        }

        Commands::AcrossSimilar { data_file_1, data_file_2, cache_dir, length_threshold, num_threads } => {
            cmd_across_similar(data_file_1,
                               data_file_2,
                               cache_dir,
                               *length_threshold,
                               *num_threads)?;
        }

        Commands::Merge { suffix_path, merged_dir, num_threads, hacksize } => {
            cmd_merge(suffix_path, merged_dir, *num_threads, *hacksize)?;
        }

        Commands::Collect { data_file, cache_dir, length_threshold } => {
            cmd_collect(data_file, cache_dir, *length_threshold)?;
        }
    }

    Ok(())
}
