use std::collections::VecDeque;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use aws_sdk_s3::{Client as S3Client, config::Region};
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use jsonpath_rust::JsonPathFinder;
use rayon::prelude::*;
use serde_json::Value;

use config::{Filterer, StreamConfig};

use crate::bloom_filter::BloomFilter;
use crate::config;
use crate::config::DedupeConfig;
use crate::s3_util::{download_to_file, object_size, upload_file};

#[derive(Clone)]
pub struct Shard {
    pub inputs: Vec<DocumentData>,
    pub filterer: Option<Filterer>,
    pub output: String,
}

#[derive(Clone)]
pub struct ShardOptions {
    pub input_work_dir: String,
    pub output_work_dir: String,
    pub annotate_only: bool,
    pub dedupe: DedupeConfig,
}

#[derive(Clone)]
pub struct DocumentData {
    pub doc_path: String,
    pub attribute_paths: Vec<String>,
}

pub trait PatternFilter {
    fn should_keep(&self, json: &Value) -> Result<bool, String>;
}

impl PatternFilter for Filterer {
    fn should_keep(&self, json: &Value) -> Result<bool, String> {
        let mut keep = self.include.len() == 0;
        for pattern in self.include.iter() {
            let mut finder = JsonPathFinder::from_str("{}", pattern)?;
            finder.set_json(Box::new(json.clone()));
            keep = finder.find() != Value::Null;
            if keep {
                break;
            }
        }
        if keep {
            for pattern in self.exclude.iter() {
                let mut finder = JsonPathFinder::from_str("{}", pattern)?;
                finder.set_json(Box::new(json.clone()));
                keep = finder.find() == Value::Null;
                if !keep {
                    break;
                }
            }
        }
        Ok(keep)
    }
}

impl Shard {
    pub fn split_streams(streams: &Vec<StreamConfig>) -> Result<Vec<Shard>, io::Error> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
        let s3_client = S3Client::new(&aws_config);

        let mut shards: Vec<Shard> = Vec::new();
        for stream_config in streams {
            let mut stream_shard_count = 0;
            log::info!("Computing shards for stream {}...", stream_config.name);
            let mut stream_inputs: Vec<String> = Vec::new();
            for pattern in &stream_config.documents {
                let index = pattern.chars().position(|c| c == '*').unwrap();
                let prefix = pattern[..index].to_string();
                let suffix = pattern[index + 2..].to_string();
                let mut has_more = true;
                let mut token: Option<String> = None;
                while has_more {
                    let resp =
                        if token.is_some() {
                            rt.block_on(s3_client.list_objects_v2()
                                .bucket("ai2-llm")
                                .prefix(&prefix)
                                .delimiter("/")
                                .continuation_token(token.unwrap())
                                .send()).unwrap()
                        } else {
                            rt.block_on(s3_client.list_objects_v2()
                                .bucket("ai2-llm")
                                .prefix(&prefix)
                                .delimiter("/")
                                .send()).unwrap()
                        };
                    for sub_folder in resp.common_prefixes().unwrap_or_default() {
                        let mut full_path = sub_folder.prefix().unwrap().to_owned();
                        full_path.push_str(&suffix);
                        stream_inputs.push(full_path);
                    }
                    token = resp.next_continuation_token().map(String::from);
                    has_more = token.is_some();
                }
            }
            stream_inputs.sort();
            let inputs_with_sizes = stream_inputs.par_iter().map(|input| {
                let resp = rt.block_on(object_size(&s3_client, "ai2-llm", input));
                let mut attr_paths = Vec::new();
                for prefix in stream_config.attributes.iter() {
                    let mut attr_prefix = "/attributes/".to_owned();
                    attr_prefix.push_str(prefix);
                    attr_prefix.push_str("/");
                    let attr_path = input.to_owned().replace("/documents/", &attr_prefix);
                    attr_paths.push(attr_path);
                }
                match resp {
                    Ok(size) =>
                        (DocumentData {
                            doc_path: input.to_owned(),
                            attribute_paths: attr_paths,
                        }, size),
                    Err(_) => {
                        (DocumentData {
                            doc_path: input.to_owned(),
                            attribute_paths: attr_paths,
                        }, 0)
                    }
                }
            }).collect::<Vec<(DocumentData, usize)>>();
            let mut shard_size = inputs_with_sizes[0].1;
            let mut shard_inputs: Vec<DocumentData> = Vec::new();
            shard_inputs.push(inputs_with_sizes[0].0.clone());
            for (input, size) in inputs_with_sizes[1..].iter() {
                if *size == 0 {
                    continue;
                }
                shard_size += size;
                if shard_size > stream_config.output.max_size_in_bytes {
                    let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, stream_shard_count);
                    let shard = Shard {
                        inputs: shard_inputs.clone(),
                        filterer: stream_config.filterer.clone(),
                        output: output.clone(),
                    };
                    shards.push(shard);
                    stream_shard_count += 1;
                    shard_size = 0;
                    shard_inputs = Vec::new();
                }
                shard_inputs.push(input.clone());
            }
            if shard_inputs.len() > 0 {
                let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, stream_shard_count);
                let shard = Shard {
                    inputs: shard_inputs.clone(),
                    filterer: stream_config.filterer.clone(),
                    output: output.clone(),
                };
                shards.push(shard);
                stream_shard_count += 1;
            }
            log::info!("Splitting {} files for {} into {} shards", stream_inputs.len(), stream_config.name, stream_shard_count);
        }

        Ok(shards)
    }
}

pub fn process_shard(
    shard: Shard,
    bloom_filter: Arc<Option<BloomFilter>>,
    options: ShardOptions,
) -> Result<(), io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();

    let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
    let s3_client = S3Client::new(&aws_config);

    let inputs_dir = Path::new(&options.input_work_dir);
    let outputs_dir = Path::new(&options.output_work_dir);

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
                let document_key = options.dedupe.clone().document_key.map(|k| -> String {
                    let mut finder = jsonpath_rust::JsonPathFinder::from_str("{}", &k).map_err(|e| io::Error::new(io::ErrorKind::Other, e)).unwrap();
                    finder.set_json(Box::new(data.clone()));
                    finder.find().as_array().unwrap().get(0).unwrap().as_str().unwrap().to_string()
                });

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
                        if options.annotate_only {
                            mutable_data["filtered"] = Value::Bool(true);
                        } else {
                            should_write = false;
                        }
                    }
                }
                if should_write {
                    for bf in bloom_filter.iter() {
                        match document_key {
                            Some(ref k) => {
                                let mut dedupe_key = VecDeque::with_capacity(1);
                                dedupe_key.push_back(k.as_str());
                                if bf.contains(&dedupe_key) {
                                    if options.annotate_only {
                                        mutable_data["duplicate"] = Value::Bool(true);
                                    } else {
                                        should_write = false;
                                    }
                                } else if !bf.read_only {
                                    bf.insert(&dedupe_key);
                                }
                            }
                            None => {
                                if options.dedupe.paragraphs {
                                    // Split the text into paragraphs and check each one.
                                    let paragraphs = mutable_data["text"].as_str().unwrap().split("\n");
                                    if options.annotate_only {
                                        let mut duplicate_lines = Vec::new();
                                        let mut line_number = 0;
                                        for p in paragraphs {
                                            let mut dedupe_key = VecDeque::with_capacity(1);
                                            dedupe_key.push_back(p);
                                            if bf.contains(&dedupe_key) {
                                                duplicate_lines.push(Value::Number(line_number.into()));
                                            } else if !bf.read_only {
                                                bf.insert(&dedupe_key);
                                            }
                                            line_number += 1;
                                        }
                                        mutable_data["duplicate_lines"] = Value::Array(duplicate_lines);
                                    } else {
                                        let mut final_text = String::new();
                                        for p in paragraphs {
                                            let mut dedupe_key = VecDeque::with_capacity(1);
                                            dedupe_key.push_back(p);
                                            if !bf.contains(&dedupe_key) {
                                                if final_text.len() > 0 {
                                                    final_text.push_str("\n");
                                                }
                                                final_text.push_str(p);
                                                if !bf.read_only {
                                                    bf.insert(&dedupe_key);
                                                }
                                            }
                                        }
                                        mutable_data["text"] = Value::String(final_text);
                                    }
                                }
                            }
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

