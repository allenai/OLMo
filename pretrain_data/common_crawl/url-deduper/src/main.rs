mod config;
mod bloom_filter;
mod shard;

use std::collections::VecDeque;
use flate2::Compression;
use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::env;
use std::process;
use std::process::exit;
use std::sync::atomic::{AtomicU32, Ordering};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde_json;
use serde_json::Value;
use threadpool::ThreadPool;
use aws_sdk_s3::{Client as S3Client, config::Region};
use aws_sdk_s3::primitives::ByteStream;
use tokio::fs::{File as TokioFile};

use config::Config;
use bloom_filter::BloomFilter;
use shard::Shard;

async fn download_from_s3(
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

async fn upload_to_s3(
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

fn process_shard(
    shard: Shard,
    bloom_filter: Arc<BloomFilter>,
    update_bloom_filter: bool,
    annotate_only: bool,
    input_work_dir: &String,
    output_work_dir: &String,
) -> Result<(), io::Error> {
    let inputs_dir = Path::new(input_work_dir);
    let outputs_dir = Path::new(output_work_dir);

    let output_path = outputs_dir.join(shard.output.clone());
    std::fs::create_dir_all(output_path.parent().unwrap())?;

    let tmp_output_path = outputs_dir.join(shard.output.clone() + ".tmp");
    let output_file = OpenOptions::new().
        read(false).
        write(true).
        create(true).
        truncate(true).
        open(tmp_output_path.clone())?;

    let mut writer = BufWriter::with_capacity(
        1024 * 1024,
        GzEncoder::new(output_file, Compression::default()));

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build().unwrap();
    let aws_config = rt.block_on(aws_config::from_env().region(Region::new("us-east-1")).load());
    let s3_client = S3Client::new(&aws_config);

    for input_path in shard.inputs {
        log::info!("Merging {} into {}", input_path, shard.output);
        let tmp_input = inputs_dir.join(Path::new(&input_path));
        log::info!("Downloading {} to {}", input_path, tmp_input.display());
        rt.block_on(download_from_s3(
            &s3_client,
            "ai2-llm",
            &input_path,
            &tmp_input,
        ))?;
        let input_file = OpenOptions::new().
            read(true).
            write(false).
            create(false).
            open(tmp_input.clone())?;
        let reader = BufReader::with_capacity(
            1024 * 1024,
            GzDecoder::new(input_file));

        let mut line_number = 0;
        let mut lines_written = 0;
        for line in reader.lines() {
            match line {
                Ok(_) => {},
                Err(e) => {
                    log::error!("Error reading line {} of {}: {}", line_number, &input_path, e);
                    break;
                }
            }
            line_number += 1;
            let line = line?;
            let mut data: Value = serde_json::from_str(&line)?;
            let url = data["metadata"]["url"].as_str().unwrap();
            let mut url_ngram = VecDeque::with_capacity(1);
            url_ngram.push_back(url);
            let mut should_write = true;

            if bloom_filter.contains(&url_ngram) {
                if annotate_only {
                    data["duplicate"] = Value::Bool(true);
                } else {
                    should_write = false;
                }
            } else {
                if update_bloom_filter {
                    bloom_filter.insert(&url_ngram);
                }
            }

            if should_write {
                lines_written += 1;
                serde_json::to_writer(&mut writer, &data)?;
                writer.write_all(b"\n")?;
            }
        }
        std::fs::remove_file(tmp_input)?;
        log::info!("Dropped {} of {} documents from {}", line_number - lines_written, line_number, &input_path);
    }

    log::info!("Uploading {} to {}", &tmp_output_path.display(), &shard.output);
    rt.block_on(upload_to_s3(
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

    let bloom_filter = BloomFilter::initialize(&config.bloom_filter).unwrap();
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
        else {
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
                !config.bloom_filter.read_only,
                false,
                &input_work_dir,
                &output_work_dir,
            ) {
                Ok(_) => {},
                Err(e) => {
                    log::error!("Error processing {:?}: {}", shard.output, e);
                    failed_shard_count_ref.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
    }
    threadpool.join();


    if !config.bloom_filter.read_only {
        let bloom_filter_file = PathBuf::from(&config.bloom_filter.file);
        log::info!("Writing bloom filter to {:?}...", config.bloom_filter.file);
        bloom_filter.write_to_file(&bloom_filter_file).unwrap();
        log::info!("Bloom filter written.");
    }
    let failure_count = failed_shard_count_ref.fetch_add(0, Ordering::Relaxed);
    if failure_count > 0 {
        log::error!("{} shards failed to process.", failure_count);
        exit(1);
    }
    else {
        log::info!("Done!");
    }
}
