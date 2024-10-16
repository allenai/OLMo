# OLMo On The Kempner Institute HPC Cluster
This section has been added to OLMo `README` page for Kempner Community to help them with a step by step guideline to run OLMo on the HPC cluster and start exploring.
## 1. Installation on The HPC Cluster 
### 1.1. Create Conda Environment and Activate it
First create your conda environment using the following command:
```bash
conda create --prefix [path_to_your_env] python=3.10
```
Activate the environment using:
```bash
conda activate [path_to_your_env]
```
### 1.2. Install PyTorch
First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.
```bash
pip3 install torch
```
### 1.3. Install OLMo from Source Code 
To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/KempnerInstitute/OLMo.git
cd OLMo
pip install -e .[all]
```
## 2. Run OLMo on the Kempner Institute HPC Cluster
Now that we have the Conda environment ready, it's time to run OLMo. To do that, we need a config file to pass it to the training script to specify all OLMo configs and hyperparameters as well as a slurm script to submit the job on the HPC cluster. 

### 2.1. Config file
 A config file has been provided in [configs/kempner_dev/7b_Olmo.yaml](configs/kempner_dev/7b_Olmo.yaml) which enables running of a 7b OLMo model training on 4 GPUs using the `c4` data which is tokenized by `t5-base` tokenizer. You can take this config file and may adjust its different hyperparameters based on your need.

Note that you should at least modify the `wandb` section of the config file according to your `wandb` account and also setup your `wandb` account on the cluster if you haven't already. You may also simply comment out the `wandb` section on the config file if you dont wish to use `wandb` for logging.
```{code} bash
wandb:
  name: ${run_name}
  entity: <entity_name>
  project: <project_name>
```

### 2.2. Slurm Script
To run OLMo on the HPC cluster using slurm, you may use the slurm script skeleton in [scripts/kempner_dev/submit_srun.sh](scripts/kempner_dev/submit_srun.sh). This will run the 7b OLMo using 4 H100 GPUs on a single node.
Note that the following items should be updated in the above slurm script skeleton:
* Account name to use the cluster - `#SBATCH --account=<account_name>`
* Path for slurm output files - `#SBATCH --output <output_path>` and `#SBATCH --error <error_output_path>`
* Conda environment name that you just created - `conda activate </path/to/your/OLMo/conda-environment>`
* Path to the folder to save the checkpoints - `export CHECKPOINTS_PATH=</path/to/save/checkpoints`
***