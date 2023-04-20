# URL Deduper

Streams over input files and dedupes them by "metadata.url" using a Bloom Filter.

See [config/example.json](example config)

## Building

Install the Rust compiler tools
```
make install-rust
```

Build the executable:
```
make
```

## Running

```
./target/release/url-deduper config/example.json
```
