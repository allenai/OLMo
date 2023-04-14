How to run in LUMI
==================

Detailed documentation is at https://docs.lumi-supercomputer.eu.
If you are reading that, keep in mind that it is written for HPC people (who care about Fortran), not ML people (who care about Python).

## Project name

Project names are strings.
They show up everywhere.
They always look like `"project_123456789`.
Look up ours in the CSC login system.

## Partitions

LUMI has different partitions.
The "G" partition is for GPUs.
For Slurm, we do big runs with `--partition standard-g`.
We do small runs for debugging on the testing partition, with `--partition small-g`.
Runs on the small partition have a maximum runtime of 30 minutes, but it seems they don't count against our quota.

## File systems

LUMI has a lot of file systems.
These are accessible from login nodes as well as compute nodes.
The names seem arbitrary.
* Your home directory. 20GB limit. Not particularly fast. Home directories are randomly spread among five different
  lustre filesystems. Some of these have outages sometimes and then you can't log in 🤷.
* `/project/project_123456789`. 50GB limit. I use this for Singularity containers and other build debris, because
  the LUMI docs told me to do this.
* `/scratch/project_123456789`. No real limit. CSC recommended we put data there, so that's what I did. Data gets deleted after 90 days of no reads, but that's not enforced right now.
* `/pfs/lustref1/flash/project_123456789`. No real limit. Very fast. CSC recommended we use this a lot. Data gets deleted after 30 days of no reads, but that's not enforced right now. We should put checkpoints here.

## Custom software

LUMI has two different ways of installing software.
Conda isn't one of them.

The `module` system puts tools into your environment, like Conda does, but it doesn't have "environments" like
Conda.
You just `module load <tool>` whenever you need something.
There is a [tiny software library](https://lumi-supercomputer.github.io/LUMI-EasyBuild-docs/) of available tools on LUMI.
If you need something that isn't in the software library, you use EasyBuild to build it.
This is cumbersome if you have many dependencies, and we do, so I don't think we should lean on this except for
bootstrapping build tools into the environment.

Singularity is the LUMI way of running Docker containers.
Singularity containers are lighter weight than Docker, somewhere between Conda and Docker.
When you run in Singularity, the host file system is still visible, but not all drives.
Some drives have to be manually mapped in with command line flags.
I have not figured out exactly how this works.
You can convert a Docker container to a Singularity container with `singularity pull --docker-login docker://my-docker-image`.
`--docker-login` makes it ask for a password every time.
There is a way to give it passwords permanently, but I haven't figured out how.

There is a `Dockerfile.lumi` in this repo.
I converted it to a Singularity image at `/project/project_123456789/containers/llm-lumi_latest.sif`.
This image contains (should contain) everything we need to run LLM training, and also enough tools to debug a node, or just do general work on the LUMI login nodes.
For example, it has tools for downloading from S3, which I needed to get the prepared data downloaded.
This image can be recreated with `singularity pull --docker-login docker://ghcr.io/allenai/llm-lumi:latest` (after building it and uploading it to ghcr.io from a Beaker machine).

## Basic setup to work in LUMI

Here is my `~/.bashrc` file, for your copy and pasting pleasure:

```bash
# Load LUMI tools
module load LUMI/22.08 partition/G
module load systools/22.08

# Allow TERM to make backspace and other keys work properly in the terminal.
# https://unix.stackexchange.com/questions/43103/backspace-tab-not-working-in-terminal-using-ssh
export TERM=vt100

# Environment variables
export PROJECT=project_123456789
export PROJECT_DIR=/project/$PROJECT
export SCRATCH_DIR=/scratch/$PROJECT
export FLASH_DIR=/pfs/lustref1/flash/$PROJECT

# Singularity likes to write giant cache files which blow up your home directory quota, so I put it on the
# flash drive.
export SINGULARITY_CACHEDIR=$FLASH_DIR/singularity_cache
export SINGULARITY_TMPDIR=$FLASH_DIR/singularity_tmp

# EasyBuild environment variables
export SBATCH_ACCOUNT=$PROJECT
export SALLOC_ACCOUNT=$SBATCH_ACCOUNT
export EBU_USER_PREFIX=/project/$SBATCH_ACCOUNT

# For downloading things from the ai2-llm bucket.
export S3_ACCESS_KEY_ID=XXXXXXX
export S3_SECRET_ACCESS_KEY=YYYYYYY

# Other API keys for logging and metric tracking.
export WANDB_API_KEY=XXXXXXX
export LOGZIO_TOKEN=XXXXXX  # find this at https://app.logz.io/#/dashboard/settings/general

# Prints current quota information.
lumi-workspaces
```

I `git clone` the repo directly into my home directory, and I'm running from there.
That seems fine.

## How running stuff works

We run everything with slurm.

### Level 1: `sbatch`

Everything starts with `sbatch script.sh`.
This uploads `script.sh` to some magical place to be executed whenever the cluster has enough capacity.
When this time comes, `script.sh` is executed.
It runs one or more `srun` commands, which puts jobs on the compute nodes.
Those slurm script files are not ordinary bash files.
They have a bunch of extra directives at the top for slurm to read.
The login nodes, the node where `script.sh` runs, and all the compute nodes, have access to the same shared file
systems.

### Level 2: `srun`

In our run script, the `srun` part looks like this:
```bash
srun \
  --cpu-bind=${CPU_BIND} \
  --distribution=block:block \
  --kill-on-bad-exit \
  <cmd1>
```

 * `--cpu-bind` is my attempt to pair up the right GPUs with the right CPUs.
Bandwidth between CPUs and GPUs isn't all the same, so I'm trying to pin the processes that service on GPU to
the right CPUs to maximize the bandwidth.
 * `--distribution` is about trying to make sure that adjacent ranks are adjacent in the cluster, i.e., making sure
that ranks 2 and 3 are on the same node, rather than spread across nodes.
 * `--kill-on-bad-exit` makes sure that when one process dies, slurm kills all the others. By default they keep running.

### Level 3: `run_with_environment.sh`

`<cmd1>` just expands into `scripts/run_with_environment.sh <cmd2>`.
This is a script that translates various Slurm environment variables into whatever the Mosaic trainer expects.
In full, it looks like this:

```bash
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))
```

Note that the documentation tells us to set `ROCR_VISIBLE_DEVICES`, but this is wrong.
Slurm already sets this.
If we set it again, bad things happen.

### Level 4: `singularity`

`<cmd2>` expands into this:
```bash
  singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    $PROJECT_DIR/containers/llm-lumi_latest.sif \
    <cmd3>
```

 * `-B` maps paths from the host into the container. I don't know why some paths have to be mapped, but others (like the home directory) are just there.

### Level 5: Our own training script

Finally we get to run our own trainer, when `<cmd3>` expands into `python scripts/train.py configs/1.2b-c4.yaml --run_name=${SLURM_JOB_ID}`.

Notice how we're not using the MosaicML launcher.
We don't need it, since slurm already launches us.
We could probably find a solution where slurm only launches one process per node, and we let MosaicML's launcher launch more, one per GPU, but I see no need.

### Monitoring your runs

* You can see all cluster activity with `squeue`.
* You can see all of your own cluster activity with `squeue --me`.
* You can see all of our project's cluster activity with `squeue -A $PROJECT`.
* You can log into a running node with `srun --interactive --pty --jobid=3265861 bash`. This will attach to the node as it runs. When the job finishes or fails, your `bash` will get killed.
* You can see the logs for a run in `${FLASH_DIR}/logs/${SLURM_JOB_ID}.log`. All nodes write to the same file. E.g. `tail -f $FLASH_DIR/logs/3376668.log`.

### Running an interactive session

This is how you can run an interactive session on a compute node:
```bash
srun --account=$PROJECT --partition=small-g --time=00:30:00 --nodes=1 --ntasks-per-node=1 --gpus-per-node=1 --pty bash
```

