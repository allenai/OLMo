import logging
import numpy as np
import os
import struct
import subprocess
import sys
import time
import torch
import torch.distributed as dist

assert sys.byteorder == 'little'

log = logging.getLogger(__name__)

class InfiniGramEngine:

    def __init__(self, cfg, eos_token_id, max_batch_size_per_device, max_seq_len, local_rank, global_rank, local_world_size, world_size):

        log.info(f'[infini-gram] Initializing engine ...')

        self.cfg = cfg
        self.max_batch_size_per_device = max_batch_size_per_device
        self.max_seq_len = max_seq_len
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.local_world_size = local_world_size
        self.world_size = world_size
        self.nnodes = world_size // local_world_size

        if cfg.sharded:
            self.group_by_lr = []
            for lr in range(local_world_size):
                group = dist.new_group(ranks=list(range(lr, world_size, local_world_size)), backend='gloo', use_local_synchronization=True)
                self.group_by_lr.append(group)
            self.group = self.group_by_lr[local_rank]
            self.rank_in_group = global_rank // local_world_size
            if os.path.exists(os.path.join(cfg.index_dir, f'{self.rank_in_group}')):
                cfg.index_dir = os.path.join(cfg.index_dir, f'{self.rank_in_group}')

        fifo_query_path = f'/tmp/infini_gram_query_{local_rank}'
        if os.path.exists(fifo_query_path):
            os.remove(fifo_query_path)
        os.mkfifo(fifo_query_path)
        fifo_response_path = f'/tmp/infini_gram_response_{local_rank}'
        if os.path.exists(fifo_response_path):
            os.remove(fifo_response_path)
        os.mkfifo(fifo_response_path)

        if local_rank == 0:
            try:
                log.info(f'Loading index from {cfg.index_dir}')
                max_batch_size = (self.nnodes * max_batch_size_per_device) if cfg.sharded else max_batch_size_per_device
                if cfg.dtype == 'u16':
                    os.popen(f'g++ -std=c++20 -O3 -pthread -Wno-stringop-overread infini_gram/infini_gram.cpp -o infini_gram/infini_gram').read()
                else:
                    os.popen(f'g++ -std=c++20 -O3 -pthread -Wno-stringop-overread infini_gram/infini_gram.cpp -o infini_gram/infini_gram -D USE_U32').read()
                subprocess.Popen(f'./infini_gram/infini_gram {cfg.index_dir} {local_world_size} {max_batch_size} {max_seq_len} {cfg.support} {cfg.mode} {eos_token_id} >> {cfg.cpp_log_path} 2>&1', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception as e:
                log.error(f'[infini-gram] Engine failed to initialize: {e}')
                exit(1)

        self.fifo_query = open(fifo_query_path, 'wb')
        self.fifo_response = open(fifo_response_path, 'rb')

        log.info(f'[infini-gram] Engine initialized')

    def get_infgram_ntd(self, input_idss, method):
        '''
        input_idss: (B, L), device=cpu
        '''

        if self.cfg.mode == 'debug':
            print(f'[infini-gram] Size of input_idss: {input_idss.size()}')
        assert input_idss.size(0) <= self.max_batch_size_per_device
        assert input_idss.size(1) <= self.max_seq_len
        assert type(method) == int and method in [2, 5]

        start_time = time.time()

        start_time_gather = time.time()
        if self.cfg.sharded:
            # Figure out the max sequence length and pad input_idss
            seq_len = input_idss.size(1)
            all_seq_len = [0 for _ in range(self.nnodes)]
            dist.all_gather_object(all_seq_len, seq_len, group=self.group)
            max_seq_len = max(all_seq_len)
            if seq_len < max_seq_len:
                input_idss = torch.cat([input_idss, torch.zeros(input_idss.size(0), max_seq_len - seq_len, dtype=input_idss.dtype, device=input_idss.device)], dim=1)

            all_input_idss = [torch.zeros_like(input_idss) for _ in range(self.nnodes)]
            dist.all_gather(all_input_idss, input_idss, group=self.group)
            input_idss = torch.cat(all_input_idss, dim=0) # (NNODES * batch_size_per_device, L)
        latency_ms_gather = (time.time() - start_time_gather) * 1000
        log.info(f'[infini-gram] Gather lantency: {latency_ms_gather:.3f} ms')

        start_time_encode = time.time()
        B, L = input_idss.size()
        query_buf = input_idss.numpy().astype(np.uint16 if self.cfg.dtype == 'u16' else np.uint32).tobytes()
        assert len(query_buf) == B * L * (2 if self.cfg.dtype == 'u16' else 4)
        latency_ms_encode = (time.time() - start_time_encode) * 1000
        log.info(f'[infini-gram] Encode lantency: {latency_ms_encode:.3f} ms')

        start_time_write = time.time()
        self.fifo_query.write(struct.pack('<Q', B))
        self.fifo_query.write(struct.pack('<Q', L))
        self.fifo_query.write(struct.pack('<Q', self.cfg.support))
        self.fifo_query.write(struct.pack('<Q', method))
        self.fifo_query.write(struct.pack('<Q', self.cfg.min_cnt))
        self.fifo_query.write(query_buf)
        self.fifo_query.flush()
        latency_ms_write = (time.time() - start_time_write) * 1000
        log.info(f'[infini-gram] Write lantency: {latency_ms_write:.3f} ms')

        start_time_read = time.time()
        response_buf_size = B * L * self.cfg.support * (2 if self.cfg.dtype == 'u16' else 4)
        response_buf = self.fifo_response.read(response_buf_size)
        assert len(response_buf) == response_buf_size
        latency_ms_read = (time.time() - start_time_read) * 1000
        log.info(f'[infini-gram] Read lantency: {latency_ms_read:.3f} ms')

        start_time_decode = time.time()
        infgram_ntd = torch.tensor(np.frombuffer(response_buf, dtype=np.uint8).view(np.uint16 if self.cfg.dtype == 'u16' else np.uint32).astype(np.int64)).view(B, L, self.cfg.support)
        latency_ms_decode = (time.time() - start_time_decode) * 1000
        log.info(f'[infini-gram] Decode lantency: {latency_ms_decode:.3f} ms')

        start_time_scatter = time.time()
        if self.cfg.sharded:
            # # all_to_all is not available in gloo backend
            # all_infgram_ntd = torch.zeros_like(infgram_ntd) # (NNODES * batch_size_per_device, L, support)
            # dist.all_to_all(all_infgram_ntd, infgram_ntd, group=self.group)
            # infgram_ntd = torch.cat(all_infgram_ntd.chunk(self.nnodes, dim=0), dim=-1) # (batch_size_per_device, L, NNODES * support)

            infgram_ntd_by_node = list(infgram_ntd.chunk(self.nnodes, dim=0)) # [NNODES * (batch_size_per_device, L, support)]
            outs = [torch.zeros_like(infgram_ntd_by_node[0]) for _ in range(self.nnodes)]
            handles = []
            for r in range(self.nnodes):
                src_global_rank = r * self.local_world_size + self.local_rank
                scatter_list = infgram_ntd_by_node if src_global_rank == self.global_rank else None
                handle = dist.scatter(outs[r], scatter_list, src=src_global_rank, group=self.group, async_op=True)
                handles.append(handle)
            for handle in handles:
                handle.wait()
            infgram_ntd = torch.cat(outs, dim=-1) # (batch_size_per_device, L, NNODES * support)

            # truncate the padding
            infgram_ntd = infgram_ntd[:, :seq_len, :]
        latency_ms_scatter = (time.time() - start_time_scatter) * 1000
        log.info(f'[infini-gram] Scatter lantency: {latency_ms_scatter:.3f} ms')

        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        log.info(f'[infini-gram] Engine total lantency: {latency_ms:.3f} ms')

        return {
            'infgram_ntd': infgram_ntd,
            'latency_ms': latency_ms,
            'latency_ms_gather': latency_ms_gather,
            'latency_ms_encode': latency_ms_encode,
            'latency_ms_write': latency_ms_write,
            'latency_ms_read': latency_ms_read,
            'latency_ms_decode': latency_ms_decode,
            'latency_ms_scatter': latency_ms_scatter,
        }
