use std::io;
use std::path::Path;

use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::{Client as S3Client};
use tokio::fs::{File as TokioFile};


pub async fn download_to_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
) -> Result<(), io::Error> {
    let result = s3_client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    std::fs::create_dir_all(path.parent().unwrap())?;
    let mut file = TokioFile::create(path).await?;
    let mut body = result.body.into_async_read();
    tokio::io::copy(&mut body, &mut file).await?;

    Ok(())
}

pub async fn upload_file(
    s3_client: &S3Client,
    bucket: &str,
    key: &str,
    path: &Path,
) -> Result<(), io::Error> {
    s3_client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(ByteStream::from_path(path).await?)
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    Ok(())
}

pub async fn object_size(s3_client: &S3Client, bucket: &str, key: &str) -> Result<usize, io::Error> {
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

