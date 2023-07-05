use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use aws_sdk_s3::Client as S3Client;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use glob::glob;
use rayon::prelude::*;
use serde_json::Value;

use crate::s3_util;
use crate::shard::shard_config::*;

// A shard is a unit of work for the mixer.
// It is a collection of input files that are combined into a single output file.
#[derive(Clone)]
pub struct Shard {
    pub inputs: Vec<DocumentPaths>,
    pub output: String,
    pub filter: Option<FilterConfig>,
    pub span_replacements: Option<Vec<SpanReplacementConfig>>,
    pub discard_fields: Option<Vec<String>>,
}

// A collection of paths to a document file and corresponding attribute files.
#[derive(Clone)]
pub struct DocumentPaths {
    pub doc_path: String,
    pub attribute_paths: Vec<String>,
}

impl Shard {
    // Partition the input files of a stream into a set of shards.
    // Try to respect the max_size_in_bytes in the configuration, but this is approximate
    // since it doesn't account for the size of any attributes to merged,
    // or documents dropped by the filter.
    pub fn split_streams(streams: &Vec<StreamConfig>) -> Result<Vec<Shard>, io::Error> {
        let mut shards: Vec<Shard> = Vec::new();
        for stream_config in streams {
            let mut stream_shard_count = 0;
            log::info!("Computing shards for stream {}...", stream_config.name);
            let stream_inputs = find_objects_matching_patterns(&stream_config.documents)?;
            let input_count = stream_inputs.len();
            let input_sizes = get_object_sizes(&stream_inputs)?;
            let inputs_with_sizes = std::iter::zip(stream_inputs, input_sizes)
                .map(|(input, size)| {
                    let mut attr_paths = Vec::new();
                    for prefix in stream_config.attributes.iter() {
                        let mut attr_prefix = "/attributes/".to_owned();
                        attr_prefix.push_str(prefix);
                        attr_prefix.push_str("/");
                        let attr_path = input.replace("/documents/", &attr_prefix);
                        attr_paths.push(attr_path);
                    }
                    (
                        DocumentPaths {
                            doc_path: input,
                            attribute_paths: attr_paths,
                        },
                        size,
                    )
                })
                .collect::<Vec<(DocumentPaths, usize)>>();
            let mut shard_size = inputs_with_sizes[0].1;
            let mut shard_inputs: Vec<DocumentPaths> = Vec::new();
            shard_inputs.push(inputs_with_sizes[0].0.clone());
            for (input, size) in inputs_with_sizes[1..].iter() {
                if *size == 0 {
                    log::warn!(
                        "Skipping input {}. Could not determine size",
                        input.doc_path
                    );
                    continue;
                }
                shard_size += size;
                if shard_size > stream_config.output.max_size_in_bytes {
                    let output = format!(
                        "{}/{}-{:04}.json.gz",
                        stream_config.output.path, stream_config.name, stream_shard_count
                    );
                    let shard = Shard {
                        inputs: shard_inputs.clone(),
                        output: output.clone(),
                        filter: stream_config.filter.clone(),
                        span_replacements: stream_config.span_replacement.clone(),
                        discard_fields: stream_config.output.discard_fields.clone(),
                    };
                    shards.push(shard);
                    stream_shard_count += 1;
                    shard_size = *size;
                    shard_inputs = Vec::new();
                }
                shard_inputs.push(input.clone());
            }
            if shard_inputs.len() > 0 {
                let output = format!(
                    "{}/{}-{:04}.json.gz",
                    stream_config.output.path, stream_config.name, stream_shard_count
                );
                let shard = Shard {
                    inputs: shard_inputs.clone(),
                    output: output.clone(),
                    filter: stream_config.filter.clone(),
                    span_replacements: stream_config.span_replacement.clone(),
                    discard_fields: stream_config.output.discard_fields.clone(),
                };
                shards.push(shard);
                stream_shard_count += 1;
            }
            log::info!(
                "Splitting {} files for {} into {} shards",
                input_count,
                stream_config.name,
                stream_shard_count
            );
        }

        Ok(shards)
    }

    // Process a shard:
    // Read all input files sequentially,
    // Merge attributes
    // Apply filters
    // Apply span replacements
    // Upload the output file to S3.
    pub fn process(&self, work_dirs: WorkDirConfig) -> Result<(), io::Error> {
        let cache = FileCache {
            s3_client: Box::new(s3_util::new_client()?),
            work: work_dirs.clone(),
        };

        let output_path = cache.prepare_output(&self.output)?;
        {
            let output_file = OpenOptions::new()
                .read(false)
                .write(true)
                .create(true)
                .truncate(true)
                .open(output_path.clone())?;

            let mut writer = BufWriter::with_capacity(
                1024 * 1024,
                GzEncoder::new(output_file, Compression::default()),
            );

            for input_path in self.inputs.iter() {
                log::info!("Merging {} into {}", input_path.doc_path, self.output);
                let local_docs_file = cache.prepare_input(&input_path.doc_path)?;
                let mut local_attr_readers = Vec::new();
                let mut attr_reader_failure_counts = Vec::new();
                for attr in &input_path.attribute_paths {
                    let local_attr_file = cache.prepare_input(&attr)?;
                    let f = OpenOptions::new()
                        .read(true)
                        .write(false)
                        .create(false)
                        .open(&local_attr_file)?;
                    let attr_reader = BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(f));
                    local_attr_readers.push((local_attr_file, attr_reader.lines()));
                    attr_reader_failure_counts.push(0);
                }
                let input_file = OpenOptions::new()
                    .read(true)
                    .write(false)
                    .create(false)
                    .open(&local_docs_file)?;
                let reader = BufReader::with_capacity(1024 * 1024, MultiGzDecoder::new(input_file));

                let mut line_number = 0;
                let mut lines_written = 0;
                for line in reader.lines() {
                    match line {
                        Ok(_) => {}
                        Err(e) => {
                            log::error!(
                                "Error reading line {} of {}: {}",
                                line_number,
                                &input_path.doc_path,
                                e
                            );
                            break;
                        }
                    }
                    line_number += 1;
                    let line = line?;
                    let mut data: Value = serde_json::from_str(&line)?;
                    let mut attrs = serde_json::Map::new();
                    let mut attr_reader_index = 0;
                    for (_, attr_reader) in local_attr_readers.iter_mut() {
                        match attr_reader.next() {
                            Some(Ok(line)) => {
                                let attr_data: Value = serde_json::from_str(&line)?;
                                assert_eq!(
                                    attr_data["id"], data["id"],
                                    "Mismatched ids for line {} of {}: {} != {}",
                                    line_number, &input_path.doc_path, attr_data["id"], data["id"]
                                );
                                for (k, v) in attr_data["attributes"].as_object().unwrap().iter() {
                                    attrs.insert(k.clone(), v.clone());
                                }
                            }
                            Some(Err(e)) => {
                                if attr_reader_failure_counts[attr_reader_index] == 0 {
                                    log::warn!(
                                        "Error reading attributes from {} at line {}: {}",
                                        input_path.attribute_paths[attr_reader_index],
                                        line_number,
                                        e
                                    );
                                }
                                attr_reader_failure_counts[attr_reader_index] += 1;
                                break;
                            }
                            None => {
                                if attr_reader_failure_counts[attr_reader_index] == 0 {
                                    log::warn!(
                                        "Missing attributes from {} at line {}",
                                        input_path.attribute_paths[attr_reader_index],
                                        line_number
                                    );
                                }
                                attr_reader_failure_counts[attr_reader_index] += 1;
                                break;
                            }
                        }
                        attr_reader_index += 1;
                    }

                    if !attrs.is_empty() {
                        // Add to existing attributes if they exist, otherwise create them.
                        if let Value::Object(ref mut existing_attrs) = data["attributes"] {
                            for (k, v) in attrs.iter() {
                                existing_attrs.insert(k.clone(), v.clone());
                            }
                        } else {
                            data["attributes"] = Value::Object(attrs);
                        }
                    }

                    let mut should_write = true;
                    for f in self.filter.iter() {
                        if !f
                            .should_keep(&data)
                            .map_err(|s| io::Error::new(io::ErrorKind::Other, s))?
                        {
                            should_write = false;
                            break;
                        }
                    }
                    if should_write {
                        if self.span_replacements.is_some() {
                            let mut replacements = self
                                .span_replacements
                                .as_ref()
                                .unwrap()
                                .iter()
                                .flat_map(|r| r.find_spans_to_replace(&data).unwrap())
                                .collect::<Vec<SpanReplacement>>();
                            if !replacements.is_empty() {
                                replacements.sort_by(|a, b| a.start.cmp(&b.start));

                                let mut new_text = String::new();
                                let old_text = data["text"].as_str().unwrap().to_owned();
                                let mut span_index = 0;
                                let mut i = 0;
                                let mut span_start_byte_index = 0;
                                let mut chars = old_text.char_indices();
                                let mut byte_index_with_char = chars.next();
                                while byte_index_with_char.is_some() {
                                    let (byte_index, c) = byte_index_with_char.unwrap();
                                    if span_index < replacements.len() {
                                        let is_inside_span = i >= replacements[span_index].start
                                            && i < replacements[span_index].end;
                                        if i == replacements[span_index].start {
                                            span_start_byte_index = byte_index;
                                        }
                                        if !is_inside_span {
                                            if i == replacements[span_index].end {
                                                if replacements[span_index].replacement.len() > 0 {
                                                    let replacement_text = replacements[span_index]
                                                        .replacement
                                                        .to_owned()
                                                        .replace(
                                                            "{}",
                                                            old_text
                                                                [span_start_byte_index..byte_index]
                                                                .to_owned()
                                                                .as_str(),
                                                        );
                                                    new_text.push_str(&replacement_text);
                                                }
                                                while span_index < replacements.len()
                                                    && replacements[span_index].start < i
                                                {
                                                    span_index += 1;
                                                }
                                            }
                                            if span_index < replacements.len()
                                                && replacements[span_index].start == i
                                            {
                                                span_start_byte_index = byte_index;
                                            } else {
                                                new_text.push(c);
                                            }
                                        }
                                    } else {
                                        new_text.push(c);
                                    }
                                    i += 1;
                                    byte_index_with_char = chars.next();
                                }
                                if span_index < replacements.len() {
                                    if replacements[span_index].replacement.len() > 0 {
                                        let replacement_text = replacements[span_index]
                                            .replacement
                                            .to_owned()
                                            .replace(
                                                "{}",
                                                old_text[span_start_byte_index..]
                                                    .to_owned()
                                                    .as_str(),
                                            );
                                        new_text.push_str(&replacement_text);
                                    }
                                }
                                data["text"] = Value::String(new_text);
                            }
                        }
                        for f in self.discard_fields.iter().flatten() {
                            data.as_object_mut().unwrap().remove(f);
                        }
                        lines_written += 1;
                        serde_json::to_writer(&mut writer, &data)?;
                        writer.write_all(b"\n")?;
                    }
                }
                cache.finalize_input(&input_path.doc_path)?;
                for i in 0..input_path.attribute_paths.len() {
                    if attr_reader_failure_counts[i] > 0 {
                        log::warn!(
                            "Failed to read {} attributes from {}",
                            input_path.attribute_paths[i],
                            attr_reader_failure_counts[i]
                        );
                    }
                    cache.finalize_input(&input_path.attribute_paths[i])?;
                }
                log::info!(
                    "Dropped {} of {} documents from {}",
                    line_number - lines_written,
                    line_number,
                    &input_path.doc_path
                );
            }
        }
        cache.finalize_output(&self.output)?;
        Ok(())
    }
}

pub mod shard_config {
    use jsonpath_rust::JsonPathFinder;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Serialize, Deserialize, Clone)]
    pub struct StreamConfig {
        pub name: String,
        // Path to core documents
        pub documents: Vec<String>,
        // Path to auxillary attributes
        pub attributes: Vec<String>,
        // json-path-based filtering
        pub filter: Option<FilterConfig>,
        // span replacement
        pub span_replacement: Option<Vec<SpanReplacementConfig>>,
        pub output: StreamOutputConfig,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct StreamOutputConfig {
        pub path: String,
        pub max_size_in_bytes: usize,
        pub discard_fields: Option<Vec<String>>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct WorkDirConfig {
        pub input: String,
        pub output: String,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct FilterConfig {
        pub include: Vec<String>,
        pub exclude: Vec<String>,
    }

    #[derive(Serialize, Deserialize, Clone)]
    pub struct SpanReplacementConfig {
        pub span: String,
        pub min_score: f64,
        pub replacement: String,
    }

    pub struct SpanReplacement {
        pub start: usize,
        pub end: usize,
        pub replacement: String,
    }

    impl SpanReplacementConfig {
        // Search for the configured attribute name in the given json
        // Attribute must contains a list of [start, end, score] spans.
        // Return a list of spans to be replaced.
        pub fn find_spans_to_replace(&self, json: &Value) -> Result<Vec<SpanReplacement>, String> {
            let mut finder = JsonPathFinder::from_str("{}", &self.span)?;
            finder.set_json(Box::new(json.clone()));
            let spans = finder.find();
            if spans == Value::Null {
                return Ok(Vec::new());
            }
            let replacements: Vec<SpanReplacement> = spans
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|span| span.as_array().unwrap().iter())
                .filter_map(|span| {
                    let span = span.as_array().unwrap();
                    let start = span[0].as_u64().unwrap();
                    let end = span[1].as_u64().unwrap();
                    let score = span[2].as_f64().unwrap();
                    if score >= self.min_score {
                        let replacement = SpanReplacement {
                            start: start as usize,
                            end: end as usize,
                            replacement: self.replacement.clone(),
                        };
                        Some(replacement)
                    } else {
                        None
                    }
                })
                .collect::<Vec<SpanReplacement>>();
            Ok(replacements)
        }
    }

    impl FilterConfig {
        // Check the json for the existence of any element matching the configured include/exclude patterns
        // Determine whether to keep the document based on the include/exclude matches
        pub fn should_keep(&self, json: &Value) -> Result<bool, String> {
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
}

// Handles input/output files, including S3 downloads/uploads
pub struct FileCache {
    pub s3_client: Box<S3Client>,
    pub work: WorkDirConfig,
}

macro_rules! parse_s3_url {
    ($url:expr) => {{
        let url = $url;
        let parts = url[5..].split("/").collect::<Vec<&str>>();
        let bucket = parts[0];
        let key = parts[1..].join("/");
        (bucket, key)
    }};
}

macro_rules! cached_s3_location {
    ($url:expr, $dir:expr) => {{
        let (bucket, key) = parse_s3_url!($url);
        (bucket, key.clone(), Path::new($dir).join(key.clone()))
    }};
}

impl FileCache {
    // If "location" is a path to a local file that exists, return it
    // If it is an S3 URL, download the contents to the working input directory, and return the path
    pub fn prepare_input(&self, location: &str) -> Result<PathBuf, io::Error> {
        if location.starts_with("s3://") {
            let (bucket, key, path) = cached_s3_location!(location, &self.work.input);
            log::info!("Downloading {} to {}", location, path.display());
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(s3_util::download_to_file(
                &self.s3_client,
                bucket,
                &key,
                &path,
            ))?;
            Ok(path.clone())
        } else {
            let path = Path::new(location);
            if path.exists() {
                Ok(path.to_path_buf())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::Other,
                    format!("File not found: {}", location),
                ))
            }
        }
    }

    // If input was downloaded from S3, delete the local cache
    // Otherwise, do nothing
    pub fn finalize_input(&self, location: &str) -> Result<(), io::Error> {
        if location.starts_with("s3://") {
            let (_, _, path) = cached_s3_location!(location, &self.work.input);
            std::fs::remove_file(&path)?;
            Ok(())
        } else {
            Ok(())
        }
    }

    // If output is an S3 URL, return a path to a new temporary location in the working output directory
    // If it is a local path, return a ".tmp" path in the same directory
    pub fn prepare_output(&self, location: &str) -> Result<PathBuf, io::Error> {
        if location.starts_with("s3://") {
            let (_, _, path) = cached_s3_location!(location, &self.work.output);
            std::fs::create_dir_all(path.parent().unwrap())?;
            Ok(path.clone())
        } else {
            let tmp_location = location.to_owned() + ".tmp";
            let path = Path::new(tmp_location.as_str());
            std::fs::create_dir_all(path.parent().unwrap())?;
            Ok(path.to_path_buf())
        }
    }

    // If "output" is an S3 URL, upload contents from the temporary file,
    //      then replace the temporary file with an empty one as a checkpoint
    // If "output" is a local path, rename the ".tmp" file to the original name
    pub fn finalize_output(&self, location: &str) -> Result<(), io::Error> {
        if location.starts_with("s3://") {
            let (bucket, key, path) = cached_s3_location!(location, &self.work.output);
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(s3_util::upload_file(&self.s3_client, &path, &bucket, &key))?;
            std::fs::remove_file(&path)?;
            {
                // Create empty file to indicate that the shard is done.
                OpenOptions::new().create(true).write(true).open(&path)?;
            }
            Ok(())
        } else {
            let tmp_path = location.to_owned() + ".tmp";
            let tmp_path = Path::new(tmp_path.as_str());
            std::fs::rename(&tmp_path, &location)?;
            Ok(())
        }
    }
}

pub fn find_objects_matching_patterns(patterns: &Vec<String>) -> Result<Vec<String>, io::Error> {
    let s3_url_count = patterns.iter().filter(|p| p.starts_with("s3://")).count();
    if s3_url_count == 0 {
        let mut matches = Vec::new();
        for pattern in patterns.iter() {
            for entry in
                glob(pattern).expect(format! {"Invalid file pattern: {}",pattern.clone()}.as_str())
            {
                matches.push(entry.unwrap().to_str().unwrap().to_owned());
            }
        }
        Ok(matches)
    } else if s3_url_count == patterns.len() {
        let s3_client = s3_util::new_client()?;
        s3_util::find_objects_matching_patterns(&s3_client, patterns)
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Cannot mix S3 and local paths",
        ))
    }
}

// Get the size in bytes of a list of objects, either S3 urls or local file paths
pub fn get_object_sizes(locations: &Vec<String>) -> Result<Vec<usize>, io::Error> {
    let s3_url_count = locations.iter().filter(|p| p.starts_with("s3://")).count();
    if s3_url_count == 0 {
        let sizes: Vec<usize> = locations
            .par_iter()
            .map(|location| {
                let path = Path::new(location);
                let metadata = path.metadata().unwrap();
                metadata.len() as usize
            })
            .collect();
        Ok(sizes)
    } else if s3_url_count == locations.len() {
        let s3_client = s3_util::new_client()?;
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let sizes = locations
            .par_iter()
            .map(|location| {
                let (bucket, key) = parse_s3_url!(location);
                let resp = rt.block_on(s3_util::object_size(&s3_client, &bucket, &key));
                match resp {
                    Ok(size) => size,
                    Err(_) => 0,
                }
            })
            .collect();
        Ok(sizes)
    } else {
        Err(io::Error::new(
            io::ErrorKind::Other,
            "Cannot mix S3 and local paths",
        ))
    }
}
