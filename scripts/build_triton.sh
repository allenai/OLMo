#!/usr/bin/env bash
#
# Script for building triton wheels.
#
# This does NOT need to be run on LUMI or any machine with AMD GPUs, but it does need to be run
# on a Linux machine. I've tested this on cirrascale in a Beaker interactive environment.
#
# Extra (non-Python) dependencies:
# zlib1g-dev - 'sudo apt-get install zlib1g-dev'

set -eu

# Configuration options:
workdir=/tmp/triton
triton_remote=https://github.com/ROCmSoftwarePlatform/triton.git
triton_ref=triton-mlir
upload_to=s3://ai2-llm/wheels/

# Clean and prepare working directory.
echo "Preparing working directory at ${workdir}..."
rm -rf ${workdir}
mkdir -p ${workdir}
cd ${workdir}

# Clone source code.
echo "Cloning source code from ${triton_remote} @ ${triton_ref}"
git clone ${triton_remote} .
git checkout ${triton_ref}

# Build it.
echo "Building wheels..."
cd python
pip install cmake wheel
python setup.py bdist_wheel

echo "✔️ Build finished"
ls -lh dist/

# Upload wheels to s3.
echo "Uploading to ${upload_to}..."
aws s3 cp dist/*.whl ${upload_to}
echo "✔️ Done"
