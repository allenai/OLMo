use std::io;
use aws_sdk_s3::{Client as S3Client, config::Region};
use rayon::prelude::*;
use serde_json::Value;
use jsonpath_rust::JsonPathFinder;

use config::{StreamConfig, Filterer};
use crate::config;
use crate::s3_util::object_size;

#[derive(Clone)]
pub struct Shard {
    pub inputs: Vec<DocumentData>,
    pub filterer: Option<Filterer>,
    pub output: String,
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

