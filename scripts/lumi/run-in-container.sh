#!/usr/bin/env bash
# Put setup of conda in an env variable if conda is needed
if [[ ! -z "${CONDA_ENV}" ]]; then
  source /opt/miniconda3/bin/activate ${CONDA_ENV}
fi

${@}
