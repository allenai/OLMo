#!/usr/bin/env bash

set -ex

mkdir -p /root/.cache
pushd /root/.cache
<<<<<<< HEAD
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache.tar.gz" | tar -xzf -
popd
export HF_DATASETS_OFFLINE=1
=======
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v2.tar.gz" | tar -xzf -
popd
export HF_DATASETS_OFFLINE=1
>>>>>>> 1236894e (Adds a config to train the 7B in the same way)
