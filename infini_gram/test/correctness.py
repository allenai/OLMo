import os
import random
import time
import numpy as np
import torch
from transformers import AutoTokenizer
from olmo.config import InfgramConfig
from infini_gram import InfiniGramEngine

index_dir = '../hb-wolf/index/v5_dolma-v1_7-wiki_olmo'
cfg = InfgramConfig(index_dir=index_dir)
max_batch_size = 1024
max_seq_len = 4096

engine = InfiniGramEngine(cfg, eos_token_id=50279, max_batch_size_per_device=max_batch_size, max_seq_len=max_seq_len, local_rank=0, global_rank=0, local_world_size=1, world_size=1)

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False, trust_remote_code=True)

input_texts = [
    # 'Paul G. Allen School of Computer Science and Engineering',
    # 'I really love natural language processing.',
    'describe it as an opportunistic data structure as it allows compression of the',
]
input_idss = tokenizer(input_texts, padding='longest', truncation=False, return_tensors='pt')['input_ids']
print(input_idss)
print()

infgram_ntd = engine.get_infgram_ntd(input_idss, method=5)['infgram_ntd']
for input_ids, output_idss in zip(input_idss, infgram_ntd):
    for input_id, output_ids in zip(input_ids, output_idss):
        print(tokenizer._convert_id_to_token(input_id), tokenizer.convert_ids_to_tokens(output_ids))
    print()
