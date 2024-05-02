import base64
import pickle
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import safetensors.torch
import torch

from olmo.aliases import PathOrStr

__all__ = [
    "state_dict_to_safetensors_file",
    "safetensors_file_to_state_dict",
]


@dataclass(eq=True, frozen=True)
class STKey:
    keys: Tuple
    value_is_pickled: bool


def encode_key(key: STKey) -> str:
    b = pickle.dumps((key.keys, key.value_is_pickled))
    b = base64.urlsafe_b64encode(b)
    return str(b, "ASCII")


def decode_key(key: str) -> STKey:
    b = base64.urlsafe_b64decode(key)
    keys, value_is_pickled = pickle.loads(b)
    return STKey(keys, value_is_pickled)


def flatten_dict(d: Dict) -> Dict[STKey, torch.Tensor]:
    result = {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[STKey((key,), False)] = value
        elif isinstance(value, dict):
            value = flatten_dict(value)
            for inner_key, inner_value in value.items():
                result[STKey((key,) + inner_key.keys, inner_key.value_is_pickled)] = inner_value
        else:
            pickled = bytearray(pickle.dumps(value))
            pickled_tensor = torch.frombuffer(pickled, dtype=torch.uint8)
            result[STKey((key,), True)] = pickled_tensor
    return result


def unflatten_dict(d: Dict[STKey, torch.Tensor]) -> Dict:
    result: Dict = {}

    for key, value in d.items():
        if key.value_is_pickled:
            value = pickle.loads(value.numpy().data)

        target_dict = result
        for k in key.keys[:-1]:
            new_target_dict = target_dict.get(k)
            if new_target_dict is None:
                new_target_dict = {}
                target_dict[k] = new_target_dict
            target_dict = new_target_dict
        target_dict[key.keys[-1]] = value

    return result


def state_dict_to_safetensors_file(state_dict: Dict, filename: PathOrStr):
    state_dict = flatten_dict(state_dict)
    state_dict = {encode_key(k): v for k, v in state_dict.items()}
    safetensors.torch.save_file(state_dict, filename)


def safetensors_file_to_state_dict(filename: PathOrStr, map_location: Optional[str] = None) -> Dict:
    if map_location is None:
        map_location = "cpu"
    state_dict = safetensors.torch.load_file(filename, device=map_location)
    state_dict = {decode_key(k): v for k, v in state_dict.items()}
    return unflatten_dict(state_dict)
