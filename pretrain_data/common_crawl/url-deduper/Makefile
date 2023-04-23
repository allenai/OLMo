url-deduper:
	cargo build --release

debug:
	cargo build

install-rust: build-tools
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

build-tools:
	sudo apt-get install --yes build-essential cmake
