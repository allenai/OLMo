mixer:
	cargo build --release

test:
	cargo test -- --nocapture

debug:
	cargo build

install-rust: build-tools
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

build-tools:
	sudo apt-get install --yes build-essential cmake
