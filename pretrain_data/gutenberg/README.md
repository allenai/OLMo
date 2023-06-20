# Gutenberg

Curator: @soldni

## Installation

First, install BerkeleyDB. On Ubuntu, this can be done with:

```bash
sudo apt-get install libdb++-dev
```

On macOS, this can be done with:

```bash
brew install berkeley-db4
export LDFLAGS="-L/opt/homebrew/opt/berkeley-db@4/lib"
export CPPFLAGS="-I/opt/homebrew/opt/berkeley-db@4/include"
```

then, install dependencies:

```bash
pip install -r requirements.txt
```
