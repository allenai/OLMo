import json
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# llm/checkpoints/22315/step1000/model/rank_0.safetensors
ckptpath = sys.argv[1]
outpath = sys.argv[2]
share_experts = False
n_experts = 16

metapath = ckptpath.replace("rank_0.safetensors", "metadata.json")
meta = json.load(open(metapath))
tensors = {}
with safe_open(ckptpath, framework="pt", device="cpu") as f:
    for key in f.keys():
        print(key)
        tensors[key] = f.get_tensor(key)
        if "ff_proj.weight" in key:
            if share_experts:
                t, b, _, _, _ = key.split(".") # transformer.blocks.9.ff_proj.weight
                new_key = ".".join([t, b, "0", "ffn", "experts", "mlp", "w1"])
                print('SPRE', meta['tensors'][key]['shape'])
                if new_key not in tensors:
                    tensors[new_key] = tensors.pop(key)
                    meta['tensors'][new_key] = meta['tensors'].pop(key)
                else:
                    # Concatenate weights, e.g. looks like below in the end
                    # {'flattened_offsets_per_file': {'rank_0.safetensors': [0, 134217728]}, 'shape': [65536, 2048], 'is_sharded': False, 'dtype': 'F32'}
                    tensors[new_key] = torch.cat([tensors[new_key], tensors.pop(key)], dim=0)
                    meta['tensors'][new_key]['shape'][0] += meta['tensors'][key]['shape'][0]
                    #print(meta['tensors'][key]['shape'][0])
                    meta['tensors'][new_key]['flattened_offsets_per_file']['rank_0.safetensors'][-1] += meta['tensors'][key]['flattened_offsets_per_file']['rank_0.safetensors'][-1]
                    meta['tensors'].pop(key)
                print('SPOST', tensors[new_key].shape, meta['tensors'][new_key]['shape'])
            else:
                new_key = key.replace("ff_proj.weight", "ffn.experts.mlp.w1")
                tensors[new_key] = tensors.pop(key)
                meta['tensors'][new_key] = meta['tensors'].pop(key)

        elif ("ff_out.weight" in key) and (key != 'transformer.ff_out.weight'):
            if share_experts:
                t, b, _, _, _ = key.split(".") # transformer.blocks.9.ff_proj.weight
                new_key = ".".join([t, b, "0", "ffn", "experts", "mlp", "w2"])
                print('SHPP Pre', meta['tensors'][key]['shape'])
                if new_key not in tensors:
                    base_shape = tensors[key].shape
                    tensors[new_key] = tensors.pop(key).reshape(meta['tensors'][key]['shape']).t().reshape(base_shape)
                    meta['tensors'][new_key] = meta['tensors'].pop(key)
                    assert len(meta['tensors'][new_key]['shape']) == 2
                    meta['tensors'][new_key]['shape'] = [meta['tensors'][new_key]['shape'][1], meta['tensors'][new_key]['shape'][0]]
                else:
                    # Concatenate weights, e.g. looks like below in the end
                    # {'flattened_offsets_per_file': {'rank_0.safetensors': [0, 134217728]}, 'shape': [65536, 2048], 'is_sharded': False, 'dtype': 'F32'}
                    tensors[new_key] = torch.cat([tensors[new_key], tensors.pop(key)], dim=0)
                    meta['tensors'][new_key]['shape'][0] += meta['tensors'][key]['shape'][1]
                    meta['tensors'][new_key]['flattened_offsets_per_file']['rank_0.safetensors'][-1] += meta['tensors'][key]['flattened_offsets_per_file']['rank_0.safetensors'][-1]
                    meta['tensors'].pop(key)
                print('SHPP Post', tensors[new_key].shape, meta['tensors'][new_key]['shape'])
            else:
                new_key = key.replace("ff_out.weight", "ffn.experts.mlp.w2")
                # Need to swap shapes as different setup in MoE
                base_shape = tensors[key].shape
                tensors[new_key] = tensors.pop(key).reshape(meta['tensors'][key]['shape']).t().reshape(base_shape)
                meta['tensors'][new_key] = meta['tensors'].pop(key)
                assert len(meta['tensors'][new_key]['shape']) == 2
                meta['tensors'][new_key]['shape'] = [meta['tensors'][new_key]['shape'][1], meta['tensors'][new_key]['shape'][0]]

            # Add router
            router_key = key.replace("ff_out.weight", "ffn.router.layer.weight")
            # Shape: (hidden_size, num_experts)
            #tensors[router_key] = torch.ones((2048 * n_experts)).squeeze()
            tensors[router_key] = torch.nn.init.normal_(torch.ones((2048 * n_experts)).squeeze(), std=0.02)
            prod = 1
            for x in tensors[router_key].shape: prod *= x
            meta['tensors'][router_key] = {'flattened_offsets_per_file': {'rank_0.safetensors': [0, prod]}, 'shape': [n_experts, 2048], 'is_sharded': False, 'dtype': 'F32'}

metaoutpath = outpath.replace("rank_0.safetensors", "metadata.json")
with open(metaoutpath, "w") as f:
    json.dump(meta, f)

save_file(tensors, outpath)
