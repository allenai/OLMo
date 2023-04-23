use std::io;
use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct BloomFilterConfig {
    pub file: String,
    pub size_in_bytes: usize,
    pub read_only: bool,
    pub estimated_doc_count: usize,
    pub desired_false_positive_rate: f64,
}

#[derive(Serialize, Deserialize)]
pub struct StreamOutputConfig {
    pub path: String,
    pub max_size_in_bytes: usize,
}

#[derive(Serialize, Deserialize)]
pub struct StreamConfig {
    pub name: String,
    pub documents: Vec<String>,
    pub output: StreamOutputConfig,
}


#[derive(Serialize, Deserialize)]
pub struct WorkDirConfig {
    pub input: String,
    pub output: String
}

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub streams: Vec<StreamConfig>,
    pub bloom_filter: BloomFilterConfig,
    pub processes: usize,
    pub work_dir: WorkDirConfig,
}

impl Config {
    pub fn read_from_file(path: &str) -> Result<Config, io::Error> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let config: Config = serde_json::from_reader(reader)?;
        Ok(config)
    }
}

