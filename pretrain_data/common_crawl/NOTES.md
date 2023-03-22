# Preparing CommonCrawl Data

## Installation

On a fresh EC2 machine:

```
# Mount the multi-TB EBS volume:
#
# See https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ebs-using-volumes.html
# lsblk
# sudo mkfs -t xfs /dev/nvme1n1
# sudo mkdir /data
# sudo mount /dev/nvme1n1 /data
# sudo chown ubuntu /data
 

# Set up your ssh keys for GitHub access
# chmod 777 ~/.ssh/id_rsa
git clone git@github.com:allenai/cc_net.git
cd cc_net

# Create a cc_net/data directory backed by the EBS volume
# mkdir /data/cc_net
# ln -s /data/cc_net ~/cc_net/data 

# Install
sudo apt-get update
sudo apt install make
make ai2-setup
```

## Running

```
python3 -u -m cc_net --config config/ai2/end-to-end.json --dump <YYYY-nn> 
```

## Troubleshooting

Look in `cc_net/data/logs` for the logs of sub-processes that handle the individual tasks. A `Failed job ... has not produced any output` error indicates that the process was killed, probably for running out of memory.
