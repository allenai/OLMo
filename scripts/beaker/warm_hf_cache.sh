#!/usr/bin/env bash

set -ex

mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache.tar.gz" | tar -xzf -
popd
export HF_DATASETS_OFFLINE=1
