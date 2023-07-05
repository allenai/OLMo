use std::io;
use std::path::Path;

use aws_sdk_s3::config::Region;
use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::Client as S3Client;
use tokio::fs::File as TokioFile;

pub async fn download_to_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
) -> Result<(), io::Error> {
    let result = s3_client
        .get_object()
        .bucket(bucket)
        .key(key.clone())
        .send()
        .await
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Error downloading {}: {}",
                    key,
                    e.message().unwrap_or_default()
                ),
            )
        })?;

    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut file = TokioFile::create(path).await?;
    let mut body = result.body.into_async_read();
    tokio::io::copy(&mut body, &mut file).await?;

    Ok(())
}

pub async fn upload_file(
    s3_client: &S3Client,
    path: &Path,
    bucket: &str,
    key: &str,
) -> Result<(), io::Error> {
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key.clone())
        .body(ByteStream::from_path(path).await?)
        .send()
        .await
        .map_err(|e| {
            io::Error::new(
                io::ErrorKind::Other,
                format!(
                    "Error uploading {}: {}",
                    key,
                    e.message().unwrap_or_default()
                ),
            )
        })?;

    Ok(())
}

pub async fn object_size(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
) -> Result<usize, io::Error> {
    let resp = s3_client
        .head_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e));
    match resp {
        Ok(resp) => Ok(resp.content_length as usize),
        Err(e) => Err(e),
    }
}

// Expand wildcard patterns into a list of object paths
// Only handles one wildcard per pattern
// e.g.: a/b/* -> a/b/1, a/b/2, a/b/3
// or:   a/*/b.txt -> a/1/b.txt, a/2/b.txt, a/3/b.txt
pub fn find_objects_matching_patterns(
    s3_client: &S3Client,
    patterns: &Vec<String>,
) -> Result<Vec<String>, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut stream_inputs: Vec<String> = Vec::new();
    for pattern in patterns.iter() {
        let start_size = stream_inputs.len();
        let mut prefix = pattern.clone();
        let mut suffix: Option<String> = Some("".to_owned());
        let maybe_index = pattern.chars().position(|c| c == '*');
        if let Some(index) = maybe_index {
            prefix = pattern[..index].to_string();
            suffix = None;
            if index < pattern.len() - 1 {
                suffix = Some(pattern[index + 2..].to_string());
            }
        }
        let mut has_more = true;
        let mut token: Option<String> = None;
        while has_more {
            let parts = prefix[5..].split("/").collect::<Vec<&str>>();
            let bucket = parts[0];
            let key = parts[1..].join("/");
            let resp = if token.is_some() {
                log::info!("Listing objects in bucket={}, prefix={}", bucket, key);
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(&key)
                        .delimiter("/")
                        .continuation_token(token.unwrap())
                        .send(),
                )
                .unwrap()
            } else {
                rt.block_on(
                    s3_client
                        .list_objects_v2()
                        .bucket(bucket)
                        .prefix(&key)
                        .delimiter("/")
                        .send(),
                )
                .unwrap()
            };
            resp.contents().unwrap_or_default().iter().for_each(|obj| {
                let s3_url = format!("s3://{}/{}", bucket, obj.key().unwrap());
                stream_inputs.push(s3_url);
            });
            suffix.iter().for_each(|s| {
                resp.common_prefixes()
                    .unwrap_or_default()
                    .iter()
                    .for_each(|sub_folder| {
                        let mut full_path = sub_folder.prefix().unwrap().to_owned();
                        full_path.push_str(s);
                        let s3_url = format!("s3://{}/{}", bucket, full_path);
                        stream_inputs.push(s3_url);
                    });
            });
            token = resp.next_continuation_token().map(String::from);
            has_more = token.is_some();
        }
        log::info!(
            "Found {} objects for pattern \"{}\"",
            stream_inputs.len() - start_size,
            pattern
        );
    }
    stream_inputs.sort();
    Ok(stream_inputs)
}

pub fn new_client() -> Result<S3Client, io::Error> {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let aws_config = rt.block_on(
        aws_config::from_env()
            .region(Region::new("us-east-1"))
            .load(),
    );
    let s3_client = S3Client::new(&aws_config);
    Ok(s3_client)
}
