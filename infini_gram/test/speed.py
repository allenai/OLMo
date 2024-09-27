import os
import random
import time
import numpy as np
import torch
from olmo.config import InfgramConfig
from infini_gram import InfiniGramEngine

index_dir = '../hb-wolf/index/v5_dolma-v1_7-wiki_olmo'
cfg = InfgramConfig(index_dir=index_dir)
max_batch_size = 1024
max_seq_len = 4096

engine = InfiniGramEngine(cfg, eos_token_id=50279, max_batch_size_per_device=max_batch_size, max_seq_len=max_seq_len, local_rank=0, global_rank=0, local_world_size=1, world_size=1)

with open(os.path.join(index_dir, 'tokenized.0'), 'rb') as f:
    ds = f.read()
tok_cnt = len(ds) // 2

for batch_size in [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:
    all_latency_ms = []
    for it in range(110):
        tok_offset = random.randint(0, tok_cnt - batch_size * max_seq_len)
        input_buf = ds[tok_offset * 2:(tok_offset + batch_size * max_seq_len) * 2]
        input_idss = np.frombuffer(input_buf, dtype=np.uint8).view(np.uint16).astype(np.int64).reshape(batch_size, max_seq_len)
        input_idss = torch.tensor(input_idss, dtype=torch.int64)
        input_idss[input_idss == 65535] = 50279
        if 'v5' in index_dir:
            input_idss = input_idss.flip(1)
        start_time = time.time()
        result = engine.get_infgram_ntd(input_idss)
        # print(f'all = {result["latency_ms"]} ms, encode = {result["latency_ms_encode"]} ms, write = {result["latency_ms_write"]} ms, read = {result["latency_ms_read"]} ms, decode = {result["latency_ms_decode"]} ms')
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        # print(f'it = {it}, latency = {latency_ms:.0f} ms')
        if it >= 10:
            all_latency_ms.append(latency_ms)
    avg_latency_ms = np.mean(all_latency_ms)
    print(f'B = {batch_size}, avg latency = {avg_latency_ms:.0f} ms')
