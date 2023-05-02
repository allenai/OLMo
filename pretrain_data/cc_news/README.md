# CC-NEWS

Data from [CC-NEWS](https://commoncrawl.org/2016/10/news-dataset-available/), a crawl of news sites with daily updates.

## Pre-processing

CC-NEWS dumps are only available in WARC format, but the CC_Net code operates on [WET files](https://commoncrawl.org/2014/04/navigating-the-warc-file-format).

Download the warc files locally:
```
make download-warc dump=2022/01
```
Run multiple times to ensure that all files have been fetched

Convert to WET format:
```
make wet dump=2022/01
```

Upload WET files:
```
make upload-wet dump=2022/01
```

## Processing with CCNET

Once uploaded, the files can be processed with the CCNET pipeline.

```
make cc-news-hashes dump=CC-NEWS/2022/01
```

```
make cc-news-transform dump=CC-NEWS/2022/01
```

## Merging/deduping

```
cd ../mixer
make
./target/release/mixer config/cc-news-v1.json
```

In practice, the URL duplication rate is 0%, so the effect of this is just to consolidate the dumps into a smaller number of files.


