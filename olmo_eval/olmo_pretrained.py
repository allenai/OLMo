import torch
from transformers import PreTrainedTokenizerFast

from olmo import Olmo, Tokenizer

class OlmoPretrained(Olmo):
    # Simple interface to make compatible with HF-style loading in Catwalk
    def __init__(self, config, init_params=True):
        super().__init__(config, init_params)

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        device_map = kwargs.get("device_map")
        device = "cuda" if device_map or torch.cuda.device_count() > 0 else "cpu"
        model = OlmoPretrained.from_checkpoint(model_path, device=device)
        device = torch.device(device)
        model.device = device
        # Don't load tokenizer for now, due to incompatible format
        tokenizer_raw = Tokenizer.from_checkpoint(model_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_raw.base_tokenizer,
            truncation=tokenizer_raw.truncate_direction,
            max_length=tokenizer_raw.truncate_to,
            eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False)
        )
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        # tokenizer.model_max_length = model.config.max_sequence_length
        # Store tokenizer directly with model
        model.tokenizer = tokenizer
        return model

    def generate(self, input_ids, max_length, max_new_tokens=None,
                 eos_token_id=None, do_sample=False, pad_token_id=None):
        max_steps = max_new_tokens or max_length - input_ids.shape[1]  # max new tokens
        return super().generate(input_ids, max_steps=max_steps, eos_token_id=eos_token_id)

