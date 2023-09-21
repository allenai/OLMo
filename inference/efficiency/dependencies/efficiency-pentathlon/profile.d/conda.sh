# All of these indications probably mean we're running on a Core Scientific
# machine, where the host UIDs and GIDs can't be resolved inside the container.
# This breaks conda, which we make sure to tell the user about.
if [[ ! -d "$HOME" || -z "$HOME" || "$HOME" == "/" ]]; then
    echo "There's no valid HOME, which means conda probably won't work as expected." \
         "You might try running this session on a Cirrascale machine instead:" \
         "https://beaker.org/clusters."
else
    # We change where conda stores environments, so that they persist across sessions.
    conda_dir="$HOME/.conda"

    # This is where conda stores information specific to each environment.
    # See: https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-environment-directories-envs-dirs
    export CONDA_ENVS_DIRS="$conda_dir/envs"
    mkdir -p "$CONDA_ENVS_DIRS"

    # This is where conda stores packages. Each environment links to packages
    # stored here, so that installing the same dependency across different environments
    # doesn't download it multiple times.
    # See: https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#specify-package-directories-pkgs-dirs
    export CONDA_PKGS_DIRS="$conda_dir/pkgs"
    mkdir -p "$CONDA_PKGS_DIRS"
fi

# Initialize conda
source /opt/miniconda3/bin/activate
eval `conda "shell.bash" "hook" 2> /dev/null`
