import cached_path
from transformers import PreTrainedTokenizerFast

NAME = 'v1_small'  # Choose between v1, v1_small, or v1_tiny
BASE_PATH = 's3://ai2-llm/tokenizer/model'
