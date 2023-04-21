# URL Deduper

Streams over input files and dedupes them by "metadata.url" using a Bloom Filter.

See [config/example.json](example config)

## Building

```
cargo build --release
```

## Running

```
./target/release/url-deduper config/example.json
```
