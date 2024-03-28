#!/bin/bash
# Run this script to bootstrap a Beaker interactive session.
#
# Most of the stuff in this script will only need to be run once per machine
# since Beaker will persist the conda environment and repo in your home directory.
# In subsequent sessions on the same machine you'll probably only have to re-export
# the environment variables and re-activate the conda environment.

set -eo pipefail

# Verify that you're authenticated with Beaker.
beaker account whoami

# Set default workspace.
beaker config set default_workspace ai2/llm-testing

# Pull a GitHub token that can be used to clone private repos.
export GITHUB_TOKEN=$(beaker secret read GITHUB_TOKEN)
export WANDB_API_KEY=$(beaker secret read WANDB_API_KEY)
export AWS_ACCESS_KEY_ID=$(beaker secret read AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(beaker secret read AWS_SECRET_ACCESS_KEY)

# Create and activate environment.
conda create -y -n LLM python=3.10
conda activate LLM
echo "conda activate LLM" >> ~/.bashrc

# Install GitHub CLI.
conda install -y gh -c conda-forge

# Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
gh auth setup-git

# Install PyTorch.
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install flash attn (and triton dependency) from our pre-built wheel.
# We need cuda dev for the old version of triton.
# NOTE: once we're able to upgrade triton to >=2.0, we can remove this.
# conda install -y -c nvidia cuda-libraries-dev=11.8 cuda-nvcc=11.8
# export CUDA_HOME="$CONDA_PREFIX"
# pip install triton==2.0.0.dev20221202 https://storage.googleapis.com/ai2-python-wheels/flash_attn/flash_attn-0.2.8%2Bcu118torch2.0.0-cp310-cp310-linux_x86_64.whl

# Check for GPUs.
python -c 'import torch; print(f"GPUs available: {torch.cuda.device_count()}")'

# Clone repo.
gh repo clone allenai/LLM
cd LLM

# Install other dependencies.
pip install -e '.[all]'
