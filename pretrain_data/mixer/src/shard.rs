use std::fs::OpenOptions;
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use flate2::Compression;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;
use rayon::prelude::*;
use serde_json::Value;

use crate::shard::shard_config::*;
use crate::s3_util;
use crate::s3_util::{download_to_file, object_size, upload_file};

// A shard is a unit of work for the mixer.
// It is a collection of input files that are combined into a single output file.
#[derive(Clone)]
pub struct Shard {
    pub inputs: Vec<DocumentPaths>,
    pub output: String,
    pub filter: Option<FilterConfig>,
    pub span_replacements: Option<Vec<SpanReplacementConfig>>,
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
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();
        let s3_client = s3_util::new_client()?;

        let mut shards: Vec<Shard> = Vec::new();
        for stream_config in streams {
            let mut stream_shard_count = 0;
            log::info!("Computing shards for stream {}...", stream_config.name);
            let stream_inputs = s3_util::find_objects_matching_patterns(&s3_client, &stream_config.documents)?;
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
                        (DocumentPaths {
                            doc_path: input.to_owned(),
                            attribute_paths: attr_paths,
                        }, size),
                    Err(_) => {
                        (DocumentPaths {
                            doc_path: input.to_owned(),
                            attribute_paths: attr_paths,
                        }, 0)
                    }
                }
            }).collect::<Vec<(DocumentPaths, usize)>>();
            let mut shard_size = inputs_with_sizes[0].1;
            let mut shard_inputs: Vec<DocumentPaths> = Vec::new();
            shard_inputs.push(inputs_with_sizes[0].0.clone());
            for (input, size) in inputs_with_sizes[1..].iter() {
                if *size == 0 {
                    log::warn!("Skipping input {}. Could not determine size", input.doc_path);
                    continue;
                }
                shard_size += size;
                if shard_size > stream_config.output.max_size_in_bytes {
                    let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, stream_shard_count);
                    let shard = Shard {
                        inputs: shard_inputs.clone(),
                        output: output.clone(),
                        filter: stream_config.filter.clone(),
                        span_replacements: stream_config.span_replacement.clone(),
                    };
                    shards.push(shard);
                    stream_shard_count += 1;
                    shard_size = *size;
                    shard_inputs = Vec::new();
                }
                shard_inputs.push(input.clone());
            }
            if shard_inputs.len() > 0 {
                let output = format!("{}/{}-{:04}.json.gz", stream_config.output.path, stream_config.name, stream_shard_count);
                let shard = Shard {
                    inputs: shard_inputs.clone(),
                    output: output.clone(),
                    filter: stream_config.filter.clone(),
                    span_replacements: stream_config.span_replacement.clone(),
                };
                shards.push(shard);
                stream_shard_count += 1;
            }
            log::info!("Splitting {} files for {} into {} shards", stream_inputs.len(), stream_config.name, stream_shard_count);
        }

        Ok(shards)
    }

    // Process a shard:
    // Read all input files sequentially,
    // Merge attributes
    // Apply filters
    // Apply span replacements
    // Upload the output file to S3.
    pub fn process(&self,
                   work_dirs: WorkDirConfig,
    ) -> Result<(), io::Error> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build().unwrap();

        let s3_client = s3_util::new_client()?;

        let inputs_dir = Path::new(&work_dirs.input);
        let outputs_dir = Path::new(&work_dirs.output);

        let output_path = outputs_dir.join(self.output.clone());
        std::fs::create_dir_all(output_path.parent().unwrap())?;

        let tmp_output_path = outputs_dir.join(self.output.clone() + ".tmp");
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


            for input_path in self.inputs.iter() {
                log::info!("Merging {} into {}", input_path.doc_path, self.output);
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
                        MultiGzDecoder::new(f));
                    local_attr_readers.push(attr_reader.lines());
                }
                let input_file = OpenOptions::new().
                    read(true).
                    write(false).
                    create(false).
                    open(local_docs_file.clone())?;
                let reader = BufReader::with_capacity(
                    1024 * 1024,
                    MultiGzDecoder::new(input_file));

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
                        // Add to existing attributes if they exist, otherwise create them.
                        if let Value::Object(ref mut existing_attrs) = mutable_data["attributes"] {
                            for (k, v) in attrs.iter() {
                                existing_attrs.insert(k.clone(), v.clone());
                            }
                        } else {
                            mutable_data["attributes"] = Value::Object(attrs);
                        }
                    }

                    let mut should_write = true;
                    for f in self.filter.iter() {
                        if !f.should_keep(&mutable_data).map_err(|s| io::Error::new(io::ErrorKind::Other, s))? {
                            should_write = false;
                            break;
                        }
                    }
                    if should_write {
                        if self.span_replacements.is_some() {
                            let mut replacements =
                                self.span_replacements.as_ref().unwrap().iter().flat_map(|r|
                                    r.find_spans_to_replace(&mutable_data).unwrap()
                                ).collect::<Vec<SpanReplacement>>();
                            if !replacements.is_empty() {
                                replacements.sort_by(|a, b|
                                    a.start.cmp(&b.start)
                                );

                                let mut new_text = String::new();
                                let old_text = mutable_data["text"].as_str().unwrap().to_owned();
                                let mut span_index = 0;
                                let mut i = 0;
                                let mut span_start_byte_index = 0;
                                let mut chars = old_text.char_indices();
                                let mut byte_index_with_char = chars.next();
                                while byte_index_with_char.is_some() {
                                    let (byte_index, c) = byte_index_with_char.unwrap();
                                    if span_index < replacements.len() {
                                        let is_inside_span =
                                            i >= replacements[span_index].start &&
                                            i < replacements[span_index].end;
                                        if i == replacements[span_index].start {
                                            span_start_byte_index = byte_index;
                                        }
                                        if !is_inside_span {
                                            if i == replacements[span_index].end {
                                                if replacements[span_index].replacement.len() > 0 {
                                                    let replacement_text =
                                                        replacements[span_index].replacement.to_owned().replace(
                                                            "{}",
                                                            old_text[span_start_byte_index..byte_index].to_owned().as_str(),
                                                        );
                                                    new_text.push_str(&replacement_text);
                                                }
                                                span_index += 1;
                                            }
                                            if span_index < replacements.len()
                                                && replacements[span_index].start == i {
                                                span_start_byte_index = byte_index;
                                            }
                                            else {
                                                new_text.push(c);
                                            }
                                        }
                                    }
                                    else {
                                        new_text.push(c);
                                    }
                                    i += 1;
                                    byte_index_with_char = chars.next();
                                }
                                if span_index < replacements.len() {
                                    if replacements[span_index].replacement.len() > 0 {
                                        let replacement_text =
                                            replacements[span_index].replacement.to_owned().replace(
                                                "{}",
                                                old_text[span_start_byte_index..].to_owned().as_str(),
                                            );
                                        new_text.push_str(&replacement_text);
                                    }
                                }
                                mutable_data["text"] = Value::String(new_text);
                            }
                        }
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

        log::info!("Uploading {} to {}", &tmp_output_path.display(), &self.output);
        rt.block_on(upload_file(
            &s3_client,
            "ai2-llm",
            &self.output,
            &tmp_output_path,
        ))?;

        {
            // Create empty file to indicate that the shard is done.
            OpenOptions::new().create(true).write(true).open(&output_path)?;
            std::fs::remove_file(&tmp_output_path)?;
        }

        Ok(())
    }
}

pub mod shard_config {
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use jsonpath_rust::JsonPathFinder;

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
            let replacements: Vec<SpanReplacement> =
                spans.as_array().unwrap().iter()
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
            }).collect::<Vec<SpanReplacement>>();
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