import os
import torch
from olmo import Olmo, ModelConfig, TrainConfig
from hf_olmo import add_hf_config_to_olmo_checkpoint

def save_dev_model(path: str):
    os.makedirs(path, exist_ok=True)

    c = TrainConfig(model=ModelConfig(d_model=4096, n_heads=16, n_layers=30, alibi=True, multi_query_attention=True, attention_layer_norm=True, include_bias=False, layer_norm_type="low_precision", max_sequence_length=2048, attention_dropout=0.0, embedding_dropout=0.0, init_device=None, mlp_ratio=8))
    o = Olmo(c.model)

    c.model.init_device = "meta"
    c.save(os.path.join(path, "config.yaml"))
    torch.save(o.state_dict(), os.path.join(path, "model.pt"))
    add_hf_config_to_olmo_checkpoint.write_config(path)

if __name__ == "__main__":
    import sys
    save_dev_model(sys.argv[1])

