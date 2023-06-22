#
# from transformers import PreTrainedTokenizerFast
# from olmo import Tokenizer
# from typing import Union, Optional, Tuple
# import os
#
#
# class OlmoTokenizerFast(PreTrainedTokenizerFast):
#
#     # TODO: olmo's tokenizer is already a wrapper around huggingface. this is potentially unnecessary.
#
#     def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
#         # TODO
#         pass
#
#     def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
#         tokenizer_raw = Tokenizer.from_checkpoint(pretrained_model_name_or_path)
#         tokenizer = cls(
#             tokenizer_object=tokenizer_raw.base_tokenizer,
#             truncation=tokenizer_raw.truncate_direction,
#             max_length=tokenizer_raw.truncate_to,
#             eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False)
#         )
#         tokenizer.model_input_names = ['input_ids', 'attention_mask']
#         return tokenizer
