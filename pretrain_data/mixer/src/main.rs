use std::env;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use serde_json;
use threadpool::ThreadPool;

use bloom_filter::BloomFilter;
use config::{Config, DedupeConfig};
use shard::{process_shard, Shard, ShardOptions};

mod config;
mod bloom_filter;
mod shard;
mod s3_util;

fn main() {
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "info")
    }
    env_logger::init();
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        log::error!("Usage: {} <config file>", args[0]);
        process::exit(1);
    }
    let config: Config = Config::read_from_file(&args[1]).unwrap();
    log::info!("Running with config: {:#?}", serde_json::to_string(&config).unwrap());
    assert!(!(config.dedupe.paragraphs && config.dedupe.document_key.is_some()), "Cannot dedupe both paragraphs and documents");

    let bloom_filter = config.bloom_filter.as_ref().map(|bloom_filter_config| {
        BloomFilter::initialize(&bloom_filter_config).unwrap()
    });
    let bloom_filter = Arc::new(bloom_filter);

    let shards = Shard::split_streams(&config.streams).unwrap();

    let threadpool = ThreadPool::new(config.processes);
    let failed_shard_count = AtomicU32::new(0);
    let failed_shard_count_ref = Arc::new(failed_shard_count);
    for shard in shards {
        let output_path = Path::new(&config.work_dir.output.clone()).join(&shard.output);
        if output_path.exists() {
            log::info!("Skipping {:?} because it already exists", shard.output);
            continue;
        }
        let shard = shard.clone();
        let bloom_filter = bloom_filter.clone();
        let input_work_dir = config.work_dir.input.clone();
        let output_work_dir = config.work_dir.output.clone();
        let dedupe: DedupeConfig = config.dedupe.clone();
        let failed_shard_count_ref = failed_shard_count_ref.clone();

        threadpool.execute(move || {
            log::info!("Building output {:?}...", shard.output);
            let options = ShardOptions {
                annotate_only: false,
                input_work_dir: input_work_dir.clone(),
                output_work_dir: output_work_dir.clone(),
                dedupe: dedupe.clone(),
            };
            match process_shard(
                shard.clone(),
                bloom_filter,
                options,
            ) {
                Ok(_) => {}
                Err(e) => {
                    log::error!("Error processing {:?}: {}", shard.output, e);
                    failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }
    threadpool.join();


    for bf in bloom_filter.iter() {
        for bf_config in config.bloom_filter.iter() {
            let bloom_filter_file = PathBuf::from(&bf_config.file);
            log::info!("Writing bloom filter to {:?}...", bf_config.file);
            bf.write_to_file(&bloom_filter_file).unwrap();
            log::info!("Bloom filter written.");
        }
    }
    let failure_count = failed_shard_count_ref.fetch_add(0, Ordering::Relaxed);
    if failure_count > 0 {
        log::error!("{} shards failed to process.", failure_count);
        process::exit(1);
    } else {
        log::info!("Done!");
    }
}

#[cfg(test)]
mod test {
    use std::fs::OpenOptions;
    use std::{env, io};
    use std::io::{BufRead, BufReader};
    use std::path::Path;
    use std::sync::Arc;

    use aws_sdk_s3::{Client as S3Client, config::Region};
    use flate2::read::GzDecoder;

    use config::DedupeConfig;
    use shard::{process_shard, Shard, ShardOptions};
    use crate::bloom_filter::BloomFilter;

    use crate::config;
    use crate::s3_util::{download_to_file, upload_file};
    use crate::shard;

    #[test]
    fn test_process_shard() -> Result<(), io::Error> {
        if env::var("RUST_LOG").is_err() {
            env::set_var("RUST_LOG", "info")
        }
        env_logger::init();


        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        let aws_config = rt.block_on(
            aws_config::from_env()
                .region(Region::new("us-east-1"))
                .load()
        );
        let s3_client = S3Client::new(&aws_config);

        rt.block_on(upload_file(&s3_client, "ai2-llm",
                                "pretraining-data/tests/mixer/inputs/documents/0000/documents.json.gz",
                                Path::new("tests/data/documents.json.gz")))?;
        rt.block_on(upload_file(&s3_client, "ai2-llm",
                                "pretraining-data/tests/mixer/inputs/attributes/pii/0000/documents.json.gz",
                                Path::new("tests/data/pii-attributes.json.gz")))?;
        rt.block_on(upload_file(&s3_client, "ai2-llm",
                                "pretraining-data/tests/mixer/inputs/attributes/toxicity/0000/documents.json.gz",
                                Path::new("tests/data/toxicity-attributes.json.gz")))?;

        let doc_data = shard::DocumentData {
            doc_path: "pretraining-data/tests/mixer/inputs/documents/0000/documents.json.gz".to_string(),
            attribute_paths: vec![
                "pretraining-data/tests/mixer/inputs/attributes/pii/0000/documents.json.gz".to_string(),
                "pretraining-data/tests/mixer/inputs/attributes/toxicity/0000/documents.json.gz".to_string(),
            ],
        };
        let filterer = config::Filterer {
            include: vec!["$.metadata[?(@.length < 10000)]".to_owned()],
            exclude: vec!["$.metadata[?(@.length < 500)]".to_owned(),
                          "$.attributes[?(@.pii)]".to_owned(),
                          "$.attributes[?(@.toxicity > 0.7)]".to_owned()],
        };
        let shard_filters = Shard {
            inputs: vec![doc_data],
            filterer: Some(filterer),
            output: "pretraining-data/tests/mixer/output.json.gz".to_string(),
        };

        fn compare_contents(expected: &str,
                            actual: &str) {
            let expected_lines = BufReader::new(OpenOptions::new().
                read(true).
                write(false).
                create(false).
                open(expected).unwrap()).lines();
            let actual_lines = BufReader::new(GzDecoder::new(OpenOptions::new().
                read(true).
                write(false).
                create(false).
                open(actual).unwrap())).lines();

            for (actual, expected) in std::iter::zip(
                expected_lines,
                actual_lines,
            ) {
                let actual = actual.unwrap();
                let expected = expected.unwrap();
                assert_eq!(actual, expected);
            }
        }

        {
            // Test mixing/filtering with annotate_only
            let options = ShardOptions {
                annotate_only: true,
                input_work_dir: "tests/work/inputs".to_owned(),
                output_work_dir: "tests/work/outputs".to_owned(),
                dedupe: DedupeConfig {
                    paragraphs: false,
                    document_key: None,
                },
            };
            process_shard(shard_filters.clone(),
                          Arc::new(None),
                          options,
            )?;

            rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                         "pretraining-data/tests/mixer/output.json.gz",
                                         Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-annotate-only.json.gz")))?;

            compare_contents("tests/data/expected/output-annotate-only.json",
                             "tests/work/outputs/pretraining-data/tests/mixer/output-annotate-only.json.gz");
        }

        {
            // Test mixing/filtering
            let options = ShardOptions {
                annotate_only: false,
                input_work_dir: "tests/work/inputs".to_owned(),
                output_work_dir: "tests/work/outputs".to_owned(),
                dedupe: DedupeConfig {
                    paragraphs: false,
                    document_key: None,
                },
            };
            process_shard(shard_filters.clone(),
                          Arc::new(None),
                          options,
            )?;

            rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                         "pretraining-data/tests/mixer/output.json.gz",
                                         Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-filter.json.gz")))?;

            compare_contents("tests/data/expected/output-filter.json",
                             "tests/work/outputs/pretraining-data/tests/mixer/output-filter.json.gz");
        }

        let shard = Shard {
            filterer: None,
            ..shard_filters.clone()
        };

        {
            // Test doc-level deduping
            let options = ShardOptions {
                annotate_only: false,
                input_work_dir: "tests/work/inputs".to_owned(),
                output_work_dir: "tests/work/outputs".to_owned(),
                dedupe: DedupeConfig {
                    paragraphs: false,
                    document_key: Some("$.metadata.cc_segment".to_owned()),
                },
            };
            let bf = BloomFilter::new(1000000, 3, false);
            process_shard(shard.clone(),
                          Arc::new(Some(bf)),
                          options,
            )?;
            rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                         "pretraining-data/tests/mixer/output.json.gz",
                                         Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-doc-dedupe.json.gz")))?;

            compare_contents("tests/data/expected/output-doc-dedupe.json",
                             "tests/work/outputs/pretraining-data/tests/mixer/output-doc-dedupe.json.gz");
        }

        {
            // Test paragraph-level deduping
            let options = ShardOptions {
                annotate_only: false,
                input_work_dir: "tests/work/inputs".to_owned(),
                output_work_dir: "tests/work/outputs".to_owned(),
                dedupe: DedupeConfig {
                    paragraphs: true,
                    document_key: None,
                },
            };
            let bf = BloomFilter::new(1000000, 3, false);
            process_shard(shard.clone(),
                          Arc::new(Some(bf)),
                          options,
            )?;
            rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                         "pretraining-data/tests/mixer/output.json.gz",
                                         Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-par-dedupe.json.gz")))?;

            compare_contents("tests/data/expected/output-par-dedupe.json",
                             "tests/work/outputs/pretraining-data/tests/mixer/output-par-dedupe.json.gz");
        }


        Ok(())
    }
}