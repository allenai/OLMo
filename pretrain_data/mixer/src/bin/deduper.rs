use std::{env, io};
use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use serde_json::{json, Value};
use threadpool::ThreadPool;

use ai2_pretraining::bloom_filter::BloomFilter;
use ai2_pretraining::s3_util;
use ai2_pretraining::s3_util::{download_to_file, upload_file};
use ai2_pretraining::shard::shard_config::WorkDirConfig;

use deduper_config::*;

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "ai2_pretraining=info,deduper=info");
    }
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        log::error!("Usage: {} <config file>", args[0]);
        process::exit(1);
    }

    let config: DeduperConfig = DeduperConfig::read_from_file(&args[1]).unwrap();
    log::info!("Running with config: {:#?}", serde_json::to_string(&config).unwrap());

    run(config);
}

pub fn run(config: DeduperConfig) {
    assert!(config.dedupe.paragraphs.is_some() ^ config.dedupe.documents.is_some(),
            "Must dedupe either paragraphs or documents");

    let s3_client = s3_util::new_client().unwrap();

    let bloom_filter = BloomFilter::initialize(&config.bloom_filter).unwrap();
    let bloom_filter = Arc::new(bloom_filter);

    let paths = s3_util::find_objects_matching_patterns(&s3_client, &config.documents).unwrap().clone();

    let threadpool = ThreadPool::new(config.processes);
    let failed_shard_count = AtomicU32::new(0);
    let failed_shard_count_ref = Arc::new(failed_shard_count);
    for p in paths {
        let path = p.clone();
        let work_dirs = config.work_dir.clone();
        let dedupe = config.dedupe.clone();
        let bloom_filter = bloom_filter.clone();
        let failed_shard_count_ref = failed_shard_count_ref.clone();
        threadpool.execute(move || {
            let result = write_attributes(
                path,
                work_dirs,
                dedupe,
                bloom_filter);
            match result {
                Ok(_) => {}
                Err(e) => {
                    log::error!("Failed to process {:?}: {}", p, e);
                    failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }
    threadpool.join();

    let bloom_filter_file = PathBuf::from(&config.bloom_filter.file);
    log::info!("Writing bloom filter to {:?}...", config.bloom_filter.file);
    bloom_filter.write_to_file(&bloom_filter_file).unwrap();
    log::info!("Bloom filter written.");
    let failure_count = failed_shard_count_ref.fetch_add(0, Ordering::Relaxed);
    if failure_count > 0 {
        log::error!("{} shards failed to process.", failure_count);
        process::exit(1);
    } else {
        log::info!("Done!");
    }
}

// Write attributes for the documents in the given file:
// For doc-level deduping, check the Bloom filter for existence of the configured key and set the configured attribute to true.
// For paragraph-level deduping, check the Bloom filter for existence of a paragraph in the text and add a span to the configured attribute.
fn write_attributes(doc_path: String,
                    work_dirs: WorkDirConfig,
                    dedupe_config: DedupeConfig,
                    bloom_filter: Arc<BloomFilter>) -> Result<(), io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let s3_client = s3_util::new_client()?;

    let input_work_dir = Path::new(&work_dirs.input);
    let output_work_dir = Path::new(&work_dirs.output);

    let output_path = {
        let mut attr_prefix = "/attributes/".to_owned();
        attr_prefix.push_str(&dedupe_config.name);
        attr_prefix.push_str("/");
        doc_path.to_owned().replace("/documents/", &attr_prefix)
    };
    let local_output = output_work_dir.join(&output_path);
    if local_output.exists() {
        log::info!("Skipping {:?} because it already exists", output_path);
        return Ok(());
    }

    std::fs::create_dir_all(local_output.parent().unwrap())?;

    let tmp_output_path = output_work_dir.join(output_path.clone() + ".tmp");
    {
        let local_input = input_work_dir.join(Path::new(&doc_path));
        log::info!("Downloading {} to {}", doc_path, local_input.display());
        rt.block_on(download_to_file(
            &s3_client,
            "ai2-llm",
            &doc_path,
            &local_input,
        ))?;
        let input_file = OpenOptions::new().
            read(true).
            write(false).
            create(false).
            open(local_input.clone())?;
        let reader = BufReader::with_capacity(
            1024 * 1024,
            MultiGzDecoder::new(input_file));

        let tmp_output = OpenOptions::new().
            read(false).
            write(true).
            create(true).
            truncate(true).
            open(&tmp_output_path)?;

        let mut writer = BufWriter::with_capacity(
            1024 * 1024,
            GzEncoder::new(tmp_output, Compression::default()));

        let mut line_number = 0;
        for line in reader.lines() {
            match line {
                Ok(_) => {}
                Err(e) => {
                    log::error!("Error reading line {} of {}: {}", line_number, &doc_path, e);
                    break;
                }
            }
            line_number += 1;
            let line = line?;
            let data: Value = serde_json::from_str(&line)?;
            let mut attributes = json!({});

            match dedupe_config.documents {
                Some(ref cfg) => {
                    let document_key = {
                        let mut finder = jsonpath_rust::JsonPathFinder::from_str("{}", &cfg.key).map_err(|e| io::Error::new(io::ErrorKind::Other, e)).unwrap();
                        finder.set_json(Box::new(data.clone()));
                        finder.find().as_array().unwrap().get(0).unwrap().as_str().unwrap().to_string()
                    };

                    let mut dedupe_key = VecDeque::with_capacity(1);
                    dedupe_key.push_back(document_key.as_str());
                    if bloom_filter.contains(&dedupe_key) {
                        attributes[&cfg.attribute_name] = Value::Bool(true);
                    } else if !bloom_filter.read_only {
                        bloom_filter.insert(&dedupe_key);
                    }
                }
                None => {}
            }
            match dedupe_config.paragraphs {
                None => {}
                Some(ref cfg) => {
                    // Split the text into paragraphs and check each one.
                    let text = data["text"].as_str().unwrap();
                    let text_length = text.len();
                    let mut offset = 0;
                    let paragraphs = text.split("\n");
                    let mut duplicate_paragraph_spans = Vec::new();
                    for p in paragraphs {
                        let par_start = offset;
                        offset += p.chars().count();
                        if offset < text_length - 1 {
                            offset += 1; // For the newline
                        }
                        let par_end = offset;

                        let mut dedupe_key = VecDeque::with_capacity(1);
                        dedupe_key.push_back(p);
                        if bloom_filter.contains(&dedupe_key) {
                            let span = vec! {Value::Number(par_start.into()), Value::Number(par_end.into()), Value::Number(1.into())};
                            duplicate_paragraph_spans.push(Value::Array(span));
                        } else if !bloom_filter.read_only {
                            bloom_filter.insert(&dedupe_key);
                        }
                    }
                    attributes[&cfg.attribute_name] = Value::Array(duplicate_paragraph_spans);
                }
            }
            let mut output_object = json!({});
            output_object["id"] = data["id"].clone();
            output_object["attributes"] = attributes;
            serde_json::to_writer(&mut writer, &output_object)?;
            writer.write_all(b"\n")?;
        }
        std::fs::remove_file(local_input)?;
    }

    log::info!("Uploading {} to {}", &tmp_output_path.display(), &output_path);
    rt.block_on(upload_file(
        &s3_client,
        "ai2-llm",
        &output_path,
        &tmp_output_path,
    ))?;

    {
        // Create empty file to indicate that the shard is done.
        OpenOptions::new().create(true).write(true).open(&local_output)?;
        std::fs::remove_file(&tmp_output_path)?;
    }

    Ok(())
}

mod deduper_config {
    use std::io;
    use std::fs::File;
    use serde::{Deserialize, Serialize};

    use ai2_pretraining::shard::shard_config::*;
    use ai2_pretraining::bloom_filter::BloomFilterConfig;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DuplicateKeyConfig {
        // Remove duplicate paragraphs
        pub paragraphs: bool,
        // Use this key to dedupe whole documents
        pub document_key: Option<String>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DocumentDedupeConfig {
        pub attribute_name: String,
        pub key: String,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct ParagraphDedupeConfig {
        pub attribute_name: String,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DedupeConfig {
        pub name: String,
        pub documents: Option<DocumentDedupeConfig>,
        pub paragraphs: Option<ParagraphDedupeConfig>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct DeduperConfig {
        pub documents: Vec<String>,
        pub work_dir: WorkDirConfig,
        pub dedupe: DedupeConfig,
        pub bloom_filter: BloomFilterConfig,
        pub processes: usize,
    }

    impl DeduperConfig {
        pub fn read_from_file(path: &str) -> Result<DeduperConfig, io::Error> {
            let file = File::open(path)?;
            let reader = io::BufReader::new(file);
            let config: DeduperConfig = serde_json::from_reader(reader)?;
            Ok(config)
        }
    }
}

#[cfg(test)]
mod test {
    use std::io;
    use std::fs::OpenOptions;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    use flate2::read::MultiGzDecoder;

    use ai2_pretraining::s3_util;
    use ai2_pretraining::s3_util::download_to_file;

    use super::*;

    fn compare_contents(expected: &str,
                        actual: &str) {
        let expected_lines = BufReader::new(
            MultiGzDecoder::new(
                OpenOptions::new().
                    read(true).
                    write(false).
                    create(false).
                    open(expected).unwrap())).lines().collect::<Vec<Result<String, io::Error>>>();
        let actual_lines = BufReader::new(
            MultiGzDecoder::new(
                OpenOptions::new().
                    read(true).
                    write(false).
                    create(false).
                    open(actual).unwrap())).lines().collect::<Vec<Result<String, io::Error>>>();

        assert_eq!(expected_lines.len(), actual_lines.len(), "Wrong number of output documents");

        for (actual, expected) in std::iter::zip(
            expected_lines,
            actual_lines,
        ) {
            let actual = actual.unwrap();
            let expected = expected.unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn test_dedupe_by_url() -> Result<(), io::Error> {
        let config = DeduperConfig::read_from_file("tests/config/dedupe-by-url.json").unwrap();
        run(config);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        let s3_client = s3_util::new_client()?;

        let local_output_file = "tests/work/output/dedupe-by-url.json.gz";
        rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                     "pretraining-data/tests/mixer/inputs/v0/attributes/dedupe_by_url/head/0000.json.gz",
                                     Path::new(local_output_file)))?;

        compare_contents("tests/data/expected/dedupe-by-url.json.gz",
                         local_output_file);
        Ok(())
    }

    #[test]
    fn test_dedupe_paragraphs() -> Result<(), io::Error> {
        let config = DeduperConfig::read_from_file("tests/config/dedupe-paragraphs.json").unwrap();
        run(config);

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        let s3_client = s3_util::new_client()?;

        let local_output_file = "tests/work/output/dedupe-paragraphs.json.gz";
        rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                     "pretraining-data/tests/mixer/inputs/v0/attributes/dedupe_paragraphs/head/0000.json.gz",
                                     Path::new(local_output_file)))?;

        compare_contents("tests/data/expected/dedupe-paragraphs.json.gz",
                         local_output_file);
        Ok(())
    }
}
