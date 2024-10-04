# Environment setup

1. Conda environment

```bash
conda env create --name <your choice> --file=environment-vista.yaml
```

2. Bash environment variables

I couldn't remember. But setting this vairables should be useful for `bitsandbytes`, `triton` (which is typically used for LoRA).

```bash
export PATH=$PATH:/opt/apps/cuda/12.4/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/apps/cuda/12.4/lib64
```
