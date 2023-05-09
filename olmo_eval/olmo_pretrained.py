import torch
from transformers import PreTrainedTokenizerFast

from olmo import Olmo, Tokenizer

class OlmoPretrained(Olmo):
    # Simple interface to make compatible with HF-style loading in Catwalk

    @classmethod
    def from_pretrained(self, model_path, **kwargs):
        device_map = kwargs.get("device_map")
        device = "cuda" if device_map or torch.cuda.device_count() > 0 else "cpu"
        model = Olmo.from_checkpoint(model_path, device=device)
        device = torch.device(device)
        model.device = device
        # Don't load tokenizer for now, due to incompatible format
        tokenizer_raw = Tokenizer.from_checkpoint(model_path)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_raw.base_tokenizer,
            truncation=tokenizer_raw.truncate_direction,
            max_length=tokenizer_raw.truncate_to,
        )
        tokenizer.model_input_names = ['input_ids', 'attention_mask']
        tokenizer.model_max_length = model.config.max_sequence_length
        # Store tokenizer directly with model
        model.tokenizer = tokenizer
        return model
