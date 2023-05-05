mod config;
mod bloom_filter;
mod shard;
mod s3_util;

use std::collections::VecDeque;
use flate2::Compression;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::env;
use std::process;
use std::sync::atomic::{AtomicU32, Ordering};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde_json;
use serde_json::Value;
use threadpool::ThreadPool;
use aws_sdk_s3::{Client as S3Client, config::Region};

use config::Config;
use bloom_filter::BloomFilter;
use shard::{Shard, PatternFilter};
use s3_util::{download_to_file, upload_file};

fn process_shard(
    shard: Shard,
    bloom_filter: Arc<Option<BloomFilter>>,
    annotate_only: bool,
    input_work_dir: &String,
    output_work_dir: &String,
) -> Result<(), io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();

    let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
    let s3_client = S3Client::new(&aws_config);

    let inputs_dir = Path::new(input_work_dir);
    let outputs_dir = Path::new(output_work_dir);

    let output_path = outputs_dir.join(shard.output.clone());
    std::fs::create_dir_all(output_path.parent().unwrap())?;

    let tmp_output_path = outputs_dir.join(shard.output.clone() + ".tmp");
    {
        let output_file = OpenOptions::new().
            read(false).
            write(true).
            create(true).
            truncate(true).
            open(tmp_output_path.clone())?;

        let mut writer = BufWriter::with_capacity(
            1024 * 1024,
            GzEncoder::new(output_file, Compression::default()));


        for input_path in shard.inputs {
            log::info!("Merging {} into {}", input_path.doc_path, shard.output);
            let local_docs_file = inputs_dir.join(Path::new(&input_path.doc_path));
            log::info!("Downloading {} to {}", input_path.doc_path, local_docs_file.display());
            rt.block_on(download_to_file(
                &s3_client,
                "ai2-llm",
                &input_path.doc_path,
                &local_docs_file,
            ))?;
            let mut local_attr_readers = Vec::new();
            for attr in &input_path.attribute_paths {
                let local_attr_file = inputs_dir.join(Path::new(&attr));
                log::info!("Downloading {} to {}", attr, local_attr_file.display());
                rt.block_on(download_to_file(
                    &s3_client,
                    "ai2-llm",
                    &attr,
                    &local_attr_file,
                ))?;
                let f = OpenOptions::new().
                    read(true).
                    write(false).
                    create(false).
                    open(local_attr_file.clone())?;
                let attr_reader = BufReader::with_capacity(
                    1024 * 1024,
                    GzDecoder::new(f));
                local_attr_readers.push(attr_reader.lines());
            }
            let input_file = OpenOptions::new().
                read(true).
                write(false).
                create(false).
                open(local_docs_file.clone())?;
            let reader = BufReader::with_capacity(
                1024 * 1024,
                GzDecoder::new(input_file));

            let mut line_number = 0;
            let mut lines_written = 0;
            for line in reader.lines() {
                match line {
                    Ok(_) => {}
                    Err(e) => {
                        log::error!("Error reading line {} of {}: {}", line_number, &input_path.doc_path, e);
                        break;
                    }
                }
                line_number += 1;
                let line = line?;
                let data: Value = serde_json::from_str(&line)?;
                let mut mutable_data = data.clone();
                let url = data["metadata"]["url"].as_str().unwrap();
                let mut url_ngram = VecDeque::with_capacity(1);
                url_ngram.push_back(url);

                let mut attrs = serde_json::Map::new();
                for attr_reader in local_attr_readers.iter_mut() {
                    match attr_reader.next() {
                        Some(Ok(line)) => {
                            let data: Value = serde_json::from_str(&line)?;
                            assert_eq!(data["id"], mutable_data["id"], "Mismatched ids for line {} of {}: {} != {}", line_number, &input_path.doc_path, data["id"], mutable_data["id"]);
                            for (k, v) in data["attributes"].as_object().unwrap().iter() {
                                attrs.insert(k.clone(), v.clone());
                            }
                        }
                        Some(Err(e)) => {
                            log::error!("Error reading attributes for line {} of {}: {}", line_number, &input_path.doc_path, e);
                            break;
                        }
                        None => {
                            log::error!("Error reading attributes for line {} of {}: EOF", line_number, &input_path.doc_path);
                            break;
                        }
                    }
                }

                if !attrs.is_empty() {
                    mutable_data["attributes"] = Value::Object(attrs);
                }

                let mut should_write = true;
                for f in shard.filterer.iter() {
                    if !f.should_keep(&mutable_data).map_err(|s| io::Error::new(io::ErrorKind::Other, s))? {
                        if annotate_only {
                            mutable_data["filtered"] = Value::Bool(true);
                        } else {
                            should_write = false;
                        }
                    }
                }
                if should_write {
                    for bf in bloom_filter.iter() {
                        if bf.contains(&url_ngram) {
                            if annotate_only {
                                mutable_data["duplicate"] = Value::Bool(true);
                            } else {
                                should_write = false;
                            }
                        } else if !bf.read_only {
                            bf.insert(&url_ngram);
                        }
                    }
                }
                if should_write {
                    lines_written += 1;
                    serde_json::to_writer(&mut writer, &mutable_data)?;
                    writer.write_all(b"\n")?;
                }
            }
            std::fs::remove_file(local_docs_file)?;
            for attr in &input_path.attribute_paths {
                std::fs::remove_file(inputs_dir.join(Path::new(&attr)))?;
            }
            log::info!("Dropped {} of {} documents from {}", line_number - lines_written, line_number, &input_path.doc_path);
        }
    }

    log::info!("Uploading {} to {}", &tmp_output_path.display(), &shard.output);
    rt.block_on(upload_file(
        &s3_client,
        "ai2-llm",
        &shard.output,
        &tmp_output_path,
    ))?;

    {
        // Create empty file to indicate that the shard is done.
        OpenOptions::new().create(true).write(true).open(&output_path)?;
    }
    std::fs::remove_file(&tmp_output_path)?;

    Ok(())
}

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
    { log::info!("Running with config: {:#?}", serde_json::to_string(&config).unwrap()) };

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
        } else {
            log::info!("Processing {:?}...", output_path)
        }
        let shard = shard.clone();
        let bloom_filter = bloom_filter.clone();
        let input_work_dir = config.work_dir.input.clone();
        let output_work_dir = config.work_dir.output.clone();
        let failed_shard_count_ref = failed_shard_count_ref.clone();

        threadpool.execute(move || {
            log::info!("Processing {:?}...", shard.output);
            match process_shard(
                shard.clone(),
                bloom_filter,
                false,
                &input_work_dir,
                &output_work_dir,
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

#[test]
fn test_process_shard() -> Result<(), io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();
    let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
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
    let shard = Shard {
        inputs: vec![doc_data],
        filterer: Some(filterer),
        output: "pretraining-data/tests/mixer/output.json.gz".to_string(),
    };
    process_shard(shard.clone(),
                  Arc::new(None),
                  true,
                  &"tests/work/inputs".to_owned(),
                  &"tests/work/outputs".to_owned(),
    )?;

    rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                 "pretraining-data/tests/mixer/output.json.gz",
                                 Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-annotate-only.json.gz")))?;

    let expected_lines = BufReader::new(OpenOptions::new().
        read(true).
        write(false).
        create(false).
        open("tests/data/expected/output-annotate-only.json")?).lines();
    let actual_lines = BufReader::new(GzDecoder::new(OpenOptions::new().
        read(true).
        write(false).
        create(false).
        open("tests/work/outputs/pretraining-data/tests/mixer/output-annotate-only.json.gz")?)).lines();

    for (actual, expected) in std::iter::zip(
        expected_lines,
        actual_lines,
    ) {
        let actual = actual?;
        let expected = expected?;
        assert_eq!(actual, expected);
    }

    process_shard(shard.clone(),
                  Arc::new(None),
                  false,
                  &"tests/work/inputs".to_owned(),
                  &"tests/work/outputs".to_owned(),
    )?;

    rt.block_on(download_to_file(&s3_client, "ai2-llm",
                                 "pretraining-data/tests/mixer/output.json.gz",
                                 Path::new("tests/work/outputs/pretraining-data/tests/mixer/output-filter.json.gz")))?;

    let expected_lines = BufReader::new(OpenOptions::new().
        read(true).
        write(false).
        create(false).
        open("tests/data/expected/output-filter.json")?).lines();
    let actual_lines = BufReader::new(GzDecoder::new(OpenOptions::new().
        read(true).
        write(false).
        create(false).
        open("tests/work/outputs/pretraining-data/tests/mixer/output-filter.json.gz")?)).lines();

    for (actual, expected) in std::iter::zip(
        expected_lines,
        actual_lines,
    ) {
        let actual = actual?;
        let expected = expected?;
        assert_eq!(actual, expected);
    }
    Ok(())
}
