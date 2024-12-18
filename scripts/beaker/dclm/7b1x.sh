set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

CONFIG_PATH=$1
shift

# Setup Python environment.
# conda shell.bash activate base

pip install uv 
uv venv
source .venv/bin/activate 

git clone https://github.com/mlfoundations/DCLM.git
uv pip install -r DCLM/requirements.txt 

apt update
apt install cmake build-essential -y
apt install g++-9 -y
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

cd DCLM
python setup.py install
cd ..

uv pip install packaging
uv pip install ninja 

export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
uv pip install flash-attn==2.5.9.post1 --no-build-isolation

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

export NCCL_DEBUG=INFO
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_SOCKET_IFNAME=ib
# export NCCL_IB_GID_INDEX=0

cd DCLM

torchrun \
--nnodes "${NUM_NODES}:${NUM_NODES}" \
--nproc-per-node 8 \
--rdzv_id 12347 \
--rdzv_backend static \
--rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
--node_rank "${BEAKER_REPLICA_RANK}" \
--rdzv_conf 'read_timeout=420' \
-m training.train -- --scale 7b_1x_fast_2e-3_lr_5e-6_zloss --data-config /weka/oe-training-default/mattj/dclm/configs/zyphra_dlcm_dedup.json --logs /weka/oe-training-default/mattj/dclm/7b1x/zyphra_dlcm_dedup/ --attn-name torch_attn --report-to-wandb --num-checkpoints 20 --acc 4 --torchcompile
