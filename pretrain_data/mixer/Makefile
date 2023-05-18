mixer:
	cargo build --release

test:
	cargo test -- --nocapture

debug:
	cargo build

build-tools:
	which cmake || sudo apt-get install --yes build-essential cmake
	which cargo || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

install-rust:
	curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y
