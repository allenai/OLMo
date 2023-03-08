FROM rocm/pytorch:rocm5.4.1_ubuntu20.04_py3.7_pytorch_1.12.1

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

WORKDIR /stage

RUN pip install --upgrade pip

WORKDIR /app/dolma

COPY dolma/ ./dolma
COPY README.md .
COPY setup.py .
COPY requirements.txt .
COPY dev-requirements.txt .

RUN pip install --no-cache-dir -e .

COPY scripts/train.py .
COPY lumi/run_with_slurm_device.sh .
