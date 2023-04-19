import os
import re
import torch

from olmo.config import ModelConfig
from olmo.model import Olmo


class OlmoPretrained(Olmo):

    @classmethod
    def from_pretrained(self, model_path, config_file, **kwargs):
        model_config = ModelConfig.load(config_file, key="model")
        if os.path.isdir(model_path):
            checkpoints = [f for f in os.listdir(model_path) if f.endswith(".pt")]
            if not checkpoints:
                raise ValueError(f"No model checkpoints found in {model_path}!")
            if 'revision' in kwargs:
                revision = kwargs['revision']
                checkpoints = [f for f in checkpoints if revision in f]
                if not checkpoints:
                    raise ValueError(f"No model checkpoints found matching revision {revision}!")
            checkpoints.sort()
            model_file = os.path.join(model_path, checkpoints[-1])
        elif os.path.isfile(model_path):
            model_file = model_path
        else:
            raise ValueError(f"Model path {model_path} not found!")
        model = Olmo(model_config, init_params=False)
        device_map = kwargs.get("device_map", "auto" if torch.cuda.device_count() > 0 else None)
        device = "cpu" if not device_map else "cuda"
        device = torch.device(device)
        state_dict = torch.load(model_file, map_location=device)['state']['model']
        new_state_dict = {re.sub("^model\\.","",k):v for k,v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.device = device
        return model






