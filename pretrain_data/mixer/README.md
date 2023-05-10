# Mixer

Combines data from multiple sources into a unified output. Input data must be in [ai2 format](../README.md).
Output data will be in the same format.

This is a Rust implementation of the [merger](../merger) script.

## Testing
```shell
make test
```

## Building

Install the Rust compiler tools. (First time only.)
```
make install-rust
```

Build the executable:
```
make
```

## Running

```
./target/release/mixer config/example.json
```

## Configuration
See sample config file [here](config/example.json)

### streams

**name**
Prefix for output file names.

**documents**
Read from input files matching any of these patterns. Accepts a single wildcard `*` character.

**attributes**
Merge attributes from the `root` S3 prefix with names in the list.
Attribute files must follow the same path structure as the documents.

**output**
Output will be uploaded to the S3 `path`. Data will be coalesced into files no bigger than `max_size_in_bytes`. 

**filterer**
Optional content-based filtering. Default = keep everything. Documents are retained if they match any of the `include` patterns
(or if no `include` patterns are specified) AND if they match none of the `exclude` patterns. Pattern syntax is [jsonpath](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html#filters).

### work_dir
Input files are temporarily cached in the `input` directory. Output files are temporarily cached
in the `output` directory, then replaced with a zero-length file after a successful upload to S3.

### processes
Number of processes to run in parallel. Default is number of CPUs.


