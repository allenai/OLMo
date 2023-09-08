#!/bin/bash

set -eo pipefail


# Constants. #
RUNTIME_DIR="/gantry-runtime"
RESULTS_DIR="/results"
CONDA_ENV_FILE="environment.yml"
PIP_REQUIREMENTS_FILE="requirements.txt"
PYTHON_VERSION=3.9
export HF_DATASETS_CACHE="/hf_datasets"

# Ensure we have all the environment variables we need.
for env_var in "$GITHUB_REPO" "$GIT_REF"; do
    if [[ -z "$env_var" ]]; then
        echo >&2 "error: required environment variable is empty"
        exit 1
    fi
done

# Check for conda, install it if needed.
if ! command -v conda &> /dev/null; then
    echo "installing conda"
    curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x ~/miniconda.sh
    ~/miniconda.sh -b -p /opt/conda
    rm ~/miniconda.sh
fi

# Initialize conda for bash.
# See https://stackoverflow.com/a/58081608/4151392
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo "
##############################################
# [GANTRY] [1/3] Installing prerequisites... #
##############################################
"

if [[ -n "$GITHUB_TOKEN" ]]; then
    # Install GitHub CLI.
    conda install gh --channel conda-forge --yes
    
    # Configure git to use GitHub CLI as a credential helper so that we can clone private repos.
    gh auth setup-git
fi

echo "
#########################################
# [GANTRY] [2/3] Cloning source code... #
#########################################
"

mkdir -p "${RUNTIME_DIR}"
cd "${RUNTIME_DIR}"

if [[ -n "$GITHUB_TOKEN" ]]; then
    gh repo clone "$GITHUB_REPO" .
else
    git clone "https://github.com/$GITHUB_REPO" .
fi

git checkout "$GIT_REF"

echo "
###############################################
# [GANTRY] [3/3] Reconstructing Python env... #
###############################################
"

if [[ -z "$VENV_NAME" ]]; then
    VENV_NAME=venv
fi

if conda activate $VENV_NAME &>/dev/null; then
    echo "[GANTRY] Using existing conda environment '$VENV_NAME'"
    # The virtual environment already exists. Possibly update it based on an environment file.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        echo "[GANTRY] Updating environment from conda env file '$CONDA_ENV_FILE'..."
        conda env update -f "$CONDA_ENV_FILE" --yes
    fi
else
    # The virtual environment doesn't exist yet. Create it.
    if [[ -f "$CONDA_ENV_FILE" ]]; then
        # Create from the environment file.
        echo "[GANTRY] Initializing environment from conda env file '$CONDA_ENV_FILE'..."
        conda env create -n "$VENV_NAME" -f "$CONDA_ENV_FILE" 
    elif [[ -z "$PYTHON_VERSION" ]]; then
        # Create a new empty environment with the whatever the default Python version is.
        echo "[GANTRY] Initializing environment with default Python version..."
        conda create -n "$VENV_NAME" pip --yes
    else
        # Create a new empty environment with the specific Python version.
        echo "[GANTRY] Initializing environment with Python $PYTHON_VERSION..."
        conda create -n "$VENV_NAME" "python=$PYTHON_VERSION" pip --yes
    fi
    conda activate "$VENV_NAME"
fi

if [[ -z "$INSTALL_CMD" ]]; then
    # Check for a 'requirements.txt' and/or 'setup.py' file.
    if [[ -f 'setup.py' ]] && [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
        echo "[GANTRY] Installing packages from 'setup.py' and '$PIP_REQUIREMENTS_FILE'..."
        pip install . -r "$PIP_REQUIREMENTS_FILE"
    elif [[ -f 'setup.py' ]]; then
        echo "[GANTRY] Installing packages from 'setup.py'..."
        pip install .
    elif [[ -f "$PIP_REQUIREMENTS_FILE" ]]; then
        echo "[GANTRY] Installing dependencies from '$PIP_REQUIREMENTS_FILE'..."
        pip install -r "$PIP_REQUIREMENTS_FILE"
    fi
else
    echo "[GANTRY] Installing packages with given command: $INSTALL_CMD"
    eval "$INSTALL_CMD"
fi

PYTHONPATH="$(pwd)"
export PYTHONPATH

# Create directory for results.
# shellcheck disable=SC2296
mkdir -p "${RESULTS_DIR}/.gantry"


echo "
#############################
# [GANTRY] Environment info #
#############################
"

echo "Using $(python --version) from $(which python)"
echo "Packages:"
if which sed >/dev/null; then
    pip freeze | sed 's/^/- /'
else
    pip freeze
fi

echo "
#############################
# [GANTRY] Setup complete ✓ #
#############################
"

if [ -z "$LIMIT" ]; then
    LIMIT=-1
fi


# Execute the arguments to this script as commands themselves, piping output into a log file.
# shellcheck disable=SC2296

echo "Single Stream"
exec efficiency-benchmark run --task "$TASK" --limit "$LIMIT" --hf_dataset_args "$HF_DATASET_ARGS" --scenario "single_stream" --output_dir "$RESULTS_DIR" -- "$@" 2>&1 | tee "${RESULTS_DIR}/.gantry/single_stream.log"

echo "Fixed Batching"
exec efficiency-benchmark run --task "$TASK" --limit "$LIMIT" --hf_dataset_args "$HF_DATASET_ARGS" --scenario "fixed_batch" --output_dir "$RESULTS_DIR" -- "$@" 2>&1 | tee "${RESULTS_DIR}/.gantry/fixed_batch.log"

echo "Random Batching"
exec efficiency-benchmark run --task "$TASK" --limit "$LIMIT" --hf_dataset_args "$HF_DATASET_ARGS" --scenario "random_batch" --output_dir "$RESULTS_DIR" -- "$@" 2>&1 | tee "${RESULTS_DIR}/.gantry/random_batch.log"

echo "Offline"
exec efficiency-benchmark run --task "$TASK" --limit "$LIMIT" --hf_dataset_args "$HF_DATASET_ARGS" --scenario "offline" --output_dir "$RESULTS_DIR" --offline_dir "$RESULTS_DIR" -- "$@" --offline 2>&1 | tee "${RESULTS_DIR}/.gantry/offline.log"
