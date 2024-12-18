"""
Hot Ham Water souping idea:
https://drive.google.com/file/d/1nzrYWMqKIqzfieOec0qbOUfyr3cjJ6YQ/view?usp=sharing

example usage 

'''
bash
	python scripts/hot_ham_water.py 
	--checkpoint-dir /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-weka-anneal-from-928646-50B-nowup_big-number-no-whammy-2
	--config /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-weka-anneal-from-928646-50B-nowup_big-number-no-whammy-2/config.yaml
	--output /weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7-weka-anneal-from-928646-50B-nowup_big-number-no-whammy-2/hot_ham_water
	--num-soups 5 
'''

"""

import argparse
import glob
import os
import torch
import re
import yaml
from tqdm import tqdm
from collections import defaultdict
import random 

from olmo.checkpoint import build_sharded_checkpointer
from olmo.config import TrainConfig
from olmo.safetensors_util import safetensors_file_to_state_dict



# =============================================================
# =                        PARSE HELPERS                      =
# =============================================================


def collect_checkpoints(base_path):
	"""Gets checkpoints as (str, int) for where the checkpoints live [sorted on step num]"""
	sub_checks = glob.glob(os.path.join(base_path, '*'))

	# Only get the things that end in step\d+
	sub_checks = [_ for _ in sub_checks if re.search(r'step\d+', os.path.basename(_)) != None]
	step_nums = [int(re.search(r'\d+', os.path.basename(_))) for _ in sub_checks]

	return sorted(list(zip(sub_checks, step_nums)), key=lambda p: p[1])




def parse_yaml_config(yaml_config):
	""" Parses the yaml config to
	1. assert that LR is linear with no warmup 
	2. get the initial LR 
	3. get the initial checkpoint 
	"""
	config = yaml.safe_load(open(yaml_config, 'r'))
	assert config['scheduler']['name'] == 'linear_with_warmup'
	assert config['scheduler']['t_warmup'] == 0


	lr = config['optimizer']['learning_rate']
	start_point = config['load_path']
	return {'lr': lr, 
			'start_point': start_point}


def get_avg_lr(initial_lr, step_start, step_end, step_final):
	""" Gets the y-value of the line that passes through		
		(0, initial_lr), (step_final, 0)
		at the x-value of (step_start + step_end) / 2

		y = mx + b
		b = initial_lr
		m = -initial_lr / step_final
	"""

	yval = lambda x: -initial_lr * x / step_final + initial_lr
	return yval((step_start + step_end) / 2)


# =========================================================
# =                     MAIN SOUPER METHOD                =
# =========================================================


def main(checkpoint_dir, config, output, num_soups):
	# Parse and setup
	yaml_output = parse_yaml_config(config)
	checkpoints = collect_checkpoints(base_path)
	checkpoints = [(yaml_output, 0)] + checkpoints

	# Compute things we need: permutations + weights-per-checkpoint
	num_deltas = len(checkpoints) - 1

	permutations = [list(range(num_deltas))]
	for soup in num_soups:
		new_soup = permutations[0][:]
		random.shuffle(new_soup)
		permutations.append(new_soup)


	step_final = checkpoints[-1][1]
	avg_lrs = [get_avg_lr(yaml_output['lr'], 
						  checkpoints[i][1],
						  checkpoinst[i+1][1],
						  step_final)
			   for i in range(num_deltas)]

	weights_per_idx = defaultdict(float)
	for perm in permutations:
		for lr_idx, delta_idx in enumerate(perm):
			# The contribution per checkpoint is (theta[delta_idx+1]-theta[delta_idx]) / lrs[delta_idx] * lrs[lr_idx]
			lr_idx = avg_lrs[lr_idx]
			denom = avg_lrs[delta_idx]
			weights_per_idx[delta_idx + 1] += lr_idx / denom
			weights_per_idx[delta_idx] -= lr_idx / denom
	weights_per_idx = {k : v / num_soups for k,v in weights_per_idx.items()}
	weights_per_idx[0] += 1.0 # <--- start at init
	print("Checkpoint weights are...")
	print(weights_per_idx)


	# Init checkpoint from baseline and then update deltas according to their weights
	checkpoint_average : dict[str, torch.Tensor] = {}
	for (idx, (path, step_num)) in tqdm(enumerate(checkpoints), desc="Loading checkpoints", position=0):
		state_dict = load_checkpoint(path)
		if len(checkpoint_average) == 0:
			checkpoint_average = {k: torch.zeros_like(v) for k,v in state_dict.items()}

		if any(k not in state_dict for k in checkpoint_average.keys()) or any(
			k not in checkpoint_average for k in state_dict.keys()
		):
			raise ValueError(f"Checkpoint {path} has different keys")

		for k in tqdm(state_dict, desc"Summing checkpoints", position=1):
			if state_dict[k].shape != checkpoint_average[k].shape:
				raise ValueError(f"Checkpoint {path} has different shape for key {k}")
			checkpoint_average[k] += state_dict[k] * weights_per_idx[idx]

		del state_dict


    print(f"Saving averaged checkpoint to {output}")
    # save the averaged checkpoint
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_average, os.path.join(output, "model.pt"))

    print("Copying config.yaml")
    # copy the config file
    with open(config, 'r') as src_f:
    	with open(os.path.join(output, 'config.yaml'), 'w') as dst_f:
    		dst_f.write(src_f.read())
    print("Done!")



if __name__ == '__main__':

	parser = argparse.ArgumentParser(description="Hot Ham Water strategy for souping")

	parser.add_argument("--checkpoint-dir", type=str, required=True,
					    help="The directory where all checkpoints are stored")
	parser.add_argument("--config", type=str, required=True,
					    help="Path to config.yaml config file")
	parser.add_argument("--output", type=str, required=True,
						help="Name of directory where the output (model.pt, config) will live")
	parser.add_argument("--num-soups", type=int, required=True,
						help="How many soup ingredients we'll use")
	args = parser.parse_args()
	
	main(args.checkpoint_dir, args.config, args.output, args.num_soups)



