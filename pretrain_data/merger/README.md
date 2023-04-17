# Merger

This script combines data from multiple sources into a unified output. Input data must be in [../README.md](ai2 format). 
Output data will be in the same format. 

Usage:
```shell
python -m merger <config-file>
```


## Configuration
See sample config file [here](config/sample.json)

### streams

**name**
Prefix for output file names.

**documents**
Read from input files matching any of the `include` patterns under the `root` prefix.

**attributes**
Merge attributes from the `root` S3 prefix with names in the `include` list. 
Attribute files must follow the same path structure as the documents.

**sampler**
Random seed and sampling rate (fraction to keep). Optional. Default = keep everything

**filterer**
Optional content-based filtering. Default = keep everything. Documents are retained if they match any of the `include` patterns 
(or if no `include` patterns are specified) AND if they match none of the `exclude` patterns. Pattern syntax is [jsonpath](https://support.smartbear.com/alertsite/docs/monitors/api/endpoint/jsonpath.html#filters).

**output**
Local file `path` where output will be written. Output records from the same stream 
will be written into files no larger than `max_file_size`. Use specified number of `processes` (default = one per CPU) 


