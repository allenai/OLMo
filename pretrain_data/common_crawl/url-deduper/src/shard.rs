use std::io;
use aws_sdk_s3::{Client as S3Client, config::Region};
use rayon::prelude::*;

use config::StreamConfig;
use crate::config;

#[derive(Clone)]
pub struct Shard {
    pub inputs: Vec<String>,
    pub output: String,
}

async fn size_of_s3_object(s3_client: &S3Client, bucket: &str, key: &str) -> Result<usize, io::Error> {
    let resp = s3_client.head_object()
        .bucket(bucket)
        .key(key)
        .send().await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e));
    match resp {
        Ok(resp) => Ok(resp.content_length as usize),
        Err(e) => Err(e),
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
                let resp = rt.block_on(size_of_s3_object(&s3_client, "ai2-llm", input));
                match resp {
                    Ok(size) =>
                        (input.to_string(), size),
                    Err(_) => {
                        (input.to_string(), 0)
                    }
                }
            }).collect::<Vec<(String, usize)>>();
            let mut shard_size = inputs_with_sizes[0].1;
            let mut shard_inputs: Vec<String> = Vec::new();
            shard_inputs.push(inputs_with_sizes[0].0.clone());
            for (input, size) in inputs_with_sizes[1..].iter() {
                if *size == 0 {
                    continue;
                }
                shard_size += size;
                if shard_size > stream_config.output.max_size_in_bytes {
                    let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, shards.len());
                    let shard = Shard {
                        inputs: shard_inputs.clone(),
                        output: output.clone(),
                    };
                    shards.push(shard);
                    shard_size = 0;
                    shard_inputs = Vec::new();
                }
                shard_inputs.push(input.clone());
            }
            if shard_inputs.len() > 0 {
                let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, shards.len());
                let shard = Shard {
                    inputs: shard_inputs.clone(),
                    output: output.clone(),
                };
                shards.push(shard);
            }
            log::info!("Splitting {} files for {} into {} shards", stream_inputs.len(), stream_config.name, shards.len());
        }

        Ok(shards)
    }
}

