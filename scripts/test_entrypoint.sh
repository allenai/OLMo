#!/bin/bash

# Exit script if any commands fail.
set -e
set -o pipefail

# Check that the environment variables have been set correctly
for env_var in "$GITHUB_TOKEN" "$COMMIT_SHA"; do
    if [[ -z "$env_var" ]]; then
        echo >&2 "error: required environment variable $env_var is empty"
        exit 1
    fi
done

# Initialize conda for bash.
# See https://stackoverflow.com/a/58081608/4151392
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Install GitHub CLI.
conda install gh --channel conda-forge

# Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
gh auth setup-git

# Clone and install tango.
mkdir LLM && cd LLM
gh repo clone allenai/LLM .
git checkout --quiet "$COMMIT_SHA"

# Install dependencies.
pip install --upgrade pip
pip install --no-cache-dir '.[all]'

# Create directory for results.
mkdir -p /results

# Execute the arguments to this script as commands themselves, piping output into a log file.
exec "$@" 2>&1 | tee /results/out.log
