
"""
1. Unshard ckpt using `python /home/niklas/OLMoE/scripts/unshard.py /data/niklas/llm/checkpoints/23485/step954000 /data/niklas/llm/checkpoints/1b-954000-unsharded --safe-tensors --model-only`
2. Run this script via `python /home/niklas/OLMoE/scripts/sparsify_ckpt_unsharded.py /data/niklas/llm/checkpoints/1b-954000-unsharded/model.safetensors`
"""
import sys
import torch
from olmo.safetensors_util import safetensors_file_to_state_dict, state_dict_to_safetensors_file

path = sys.argv[1]
sd = safetensors_file_to_state_dict(path)
tensors = {}
swiglu = True
noise = False
n_experts = 8
D = 2048

def noise_injection(weight, noise_ratio=0.5, init_std=0.02):
    mask = torch.FloatTensor(weight.size()).uniform_() < moe_noise_ratio
    mask = mask.to(weight.device)
    rand_weight = torch.nn.init.normal_(copy.deepcopy(weight), mean=0.0, std=init_std)
    weight[mask] = rand_weight[mask]
    return weight

for key in list(sd.keys()):
    if "ff_proj.weight" in key:
        new_key = key.replace("ff_proj.weight", "ffn.experts.mlp.w1")
        if swiglu:
            new_key_v1 = new_key.replace("w1", "v1")
            # OLMo takes the F.silu on the second part of the tensor which corresponds to v1
            v1, w1 = sd.pop(key).chunk(2, dim=0) # e.g. [16384, 2048]
            tensors[new_key] = torch.cat([w1] * n_experts, dim=0)
            tensors[new_key_v1] = torch.cat([v1] * n_experts, dim=0)
            if noise:
                tensors[new_key] = noise_injection(tensors[new_key])
                tensors[new_key_v1] = noise_injection(tensors[new_key_v1])
        else:
            tensors[new_key] = torch.cat([sd.pop(key)] * n_experts, dim=0)
    elif ("ff_out.weight" in key) and (key != 'transformer.ff_out.weight'):
        new_key = key.replace("ff_out.weight", "ffn.experts.mlp.w2")
        tensors[new_key] = torch.cat([sd.pop(key).t()] * n_experts, dim=0)
        if noise:
            tensors[new_key] = noise_injection(tensors[new_key])
        # Add router
        router_key = key.replace("ff_out.weight", "ffn.router.layer.weight")
        # tensors[router_key] = torch.ones((n_experts, D)).squeeze() # Worse perf
        tensors[router_key] = torch.nn.init.normal_(torch.ones((n_experts, D)).squeeze(), std=0.02)
    else:
        tensors[key] = sd.pop(key)


state_dict_to_safetensors_file(tensors, path.replace("model.safetensors", "model_sparse.safetensors"))
