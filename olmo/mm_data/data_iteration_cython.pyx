#cython: language_level=3
"""Contains optimized versions of a few functions in `data_iteration` that are terribly slow in python"""
import collections

import numpy as np
cimport numpy as cnp

# Replicate these just so we don't have to import them from the python module,
# which would prevent use from using pyximport
DOC_ID_DTYPE = np.dtype([
    ('file_id', np.uint16),
    ('start_byte', np.uint64),
    ('length', np.uint32)
])

DOC_TOKENS_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('num_tokens', np.uint32),
])

DOC_SEQUENCE_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('sequence_number', np.uint64),
])

# Define structs matching these dtypes so we can get memory views of those array
cdef packed struct DocId:
    cnp.uint16_t file_id
    cnp.uint64_t start_byte
    cnp.uint32_t length

cdef packed struct DocInfo:
    DocId doc_id
    cnp.uint32_t num_tokens
    cnp.uint8_t pure_text

cdef packed struct DocSequence:
    DocId doc_id
    cnp.uint64_t sequence_number

cdef packed struct DocTokens:
    DocId doc_id
    cnp.uint32_t num_tokens
    

def reorder_sequence(cnp.ndarray idx_arr, cnp.ndarray new_order_arr):
    cdef DocSequence[:] idx = idx_arr
    cdef cnp.int64_t[:] new_order = new_order_arr

    cdef long long n_seq = int(idx_arr["sequence_number"][-1]) + 1
    cdef cnp.ndarray seq_counts_arr = np.zeros(n_seq, dtype=np.int64)

    cdef cnp.int64_t[:] seq_counts = seq_counts_arr
    cdef long long j
    cdef long long size = len(idx_arr)
    for j in range(size):
        seq_counts[new_order[idx[j].sequence_number]] += 1

    cdef cnp.ndarray start_arr = np.empty_like(seq_counts_arr)
    np.cumsum(seq_counts_arr[:-1], out=start_arr[1:])
    start_arr[0] = 0

    cdef cnp.ndarray out_arr = np.zeros_like(idx_arr)
    cdef cnp.int64_t[:] starts = start_arr
    cdef DocSequence[:] out = out_arr

    cdef long long on = 0
    cdef long long src_idx = 0

    cdef long long seq, old_seq, count
    for seq in range(n_seq):
        new_seq = new_order[seq]
        count = seq_counts[new_seq]
        src_idx = starts[new_seq]
        for j in range(count):
            out[src_idx+j].doc_id = idx[on].doc_id
            out[src_idx+j].sequence_number = new_seq
            on += 1
    return out_arr


def sequential(doc_arr: cnp.ndarray, long long max_seq_len):
    documents: DocInfo[:] = doc_arr
    cdef long long size = len(doc_arr)
    cdef cnp.ndarray out_arr = np.zeros(size, dtype=DOC_SEQUENCE_DTYPE)
    out_arr["doc_id"] = doc_arr["doc_id"]
    cdef DocSequence[:] out = out_arr

    cdef long long i
    cdef cnp.uint64_t sequence_number = 0
    cdef cnp.uint32_t total_tokens = 0
    cdef cnp.uint32_t num_tokens
    for i in range(size):
        num_tokens = documents[i].num_tokens
        if total_tokens + num_tokens > max_seq_len:
            sequence_number += 1
            total_tokens = 0
        total_tokens += num_tokens
        out[i].sequence_number = sequence_number
    return out_arr


def sequential_split_text(doc_arr: cnp.ndarray, cnp.uint32_t max_seq_len):
    documents: DocInfo[:] = doc_arr
    cdef long long size = len(doc_arr)

    max_splits = int(doc_arr["num_tokens"].sum()) // max_seq_len
    max_out = len(documents) + max_splits
    cdef cnp.ndarray out_arr = np.zeros(max_out, dtype=DOC_SEQUENCE_DTYPE)
    cdef DocSequence[:] out = out_arr

    # local variables in the loop
    cdef cnp.uint32_t next_len, num_tokens, tokens_to_take, n_full_length_splits
    cdef cnp.uint16_t file_id
    cdef cnp.uint64_t on_start

    # counters
    cdef cnp.uint64_t sequence_number = 0
    cdef cnp.uint32_t total_tokens = 0
    cdef long long out_ix = 0
    cdef long long on_ix = 0
    cdef long long i, j
    for i in range(size):
        num_tokens = documents[i].num_tokens
        next_len = total_tokens + num_tokens
        if next_len < max_seq_len:
            total_tokens += num_tokens
            out[out_ix].doc_id = documents[i].doc_id
            out[out_ix].sequence_number = sequence_number
            out_ix += 1
        elif next_len == max_seq_len:
            total_tokens = 0
            out[out_ix].doc_id = documents[i].doc_id
            out[out_ix].sequence_number = sequence_number
            out_ix += 1
            sequence_number += 1
        elif not documents[i].pure_text:
            sequence_number += 1
            out[out_ix].doc_id = documents[i].doc_id
            out[out_ix].sequence_number = sequence_number
            out_ix += 1
            total_tokens = num_tokens
        else:
            file_id = documents[i].doc_id.file_id
            on_start = documents[i].doc_id.start_byte

            # Write the half of the document that will complete this array
            tokens_to_take = max_seq_len - total_tokens
            out[out_ix].doc_id.file_id = file_id
            out[out_ix].doc_id.start_byte = on_start
            out[out_ix].doc_id.length = tokens_to_take*2
            out[out_ix].sequence_number = sequence_number
            on_start += tokens_to_take*2
            out_ix += 1
            sequence_number += 1
            num_tokens -= tokens_to_take

            # Write any max_seq_len copies needed
            n_full_length_splits = num_tokens // max_seq_len
            for j in range(n_full_length_splits):
                out[out_ix].doc_id.file_id = file_id
                out[out_ix].doc_id.start_byte = on_start
                out[out_ix].doc_id.length = max_seq_len*2
                out[out_ix].sequence_number = sequence_number
                on_start += max_seq_len*2
                out_ix += 1
                sequence_number += 1

            # Write the tail end of the document
            total_tokens = num_tokens - n_full_length_splits*max_seq_len
            if total_tokens > 0:
                out[out_ix].doc_id.file_id = file_id
                out[out_ix].doc_id.start_byte = on_start
                out[out_ix].doc_id.length = total_tokens*2
                out[out_ix].sequence_number = sequence_number
                out_ix += 1

    return out_arr[:out_ix]


cdef class DocumentPool:
    cdef public:
        list n_tokens_to_ex_id
        int seq_len
        int max_val
        cnp.ndarray hist
        int hist_min

    def __init__(self, int max_seq_len, int pool_size):
        self.n_tokens_to_ex_id = [collections.deque() for _ in range(max_seq_len+1)]
        self.hist = np.zeros(max_seq_len+1, dtype=np.int32)
        self.max_val = max_seq_len + 1
        self.hist_min = self.max_val
        self.seq_len = max_seq_len

    cpdef add(self, DocId doc_id, int num_tokens):
        cdef cnp.int32_t[:] hist = self.hist
        self.n_tokens_to_ex_id[num_tokens].append(doc_id)
        hist[num_tokens] += 1
        self.hist_min = min(num_tokens, self.hist_min)
        assert hist[self.hist_min] > 0

    cpdef pop(self, int seq_len):
        cdef cnp.int32_t[:] hist = self.hist
        cdef int j
        hist[seq_len] -= 1
        if seq_len == self.hist_min and hist[seq_len] == 0:
            for j in range(self.hist_min, self.max_val):
                if hist[j] > 0:
                    self.hist_min = j
                    break
            if self.hist_min == seq_len:
                self.hist_min = self.max_val
        return self.n_tokens_to_ex_id[seq_len].popleft()

    cpdef is_empty(self):
        return self.hist_min == self.max_val

    cpdef find_at_most(self, int ix):
        cdef cnp.int32_t[:] hist = self.hist
        if ix < self.hist_min:
            return 0
        else:
            while True:
                if hist[ix] > 0:
                    return ix
                ix -= 1

    cpdef get_first(self):
        cdef int j
        cdef cnp.int32_t[:] hist = self.hist
        for j in range(len(self.hist)-1, -1, -1):
            if hist[j] > 0:
                return j
        raise ValueError("Pool is empty")



def optimize_last(doc_arr: cnp.ndarray, max_seq_len: int, pool_size: int):
    assert np.all(doc_arr["num_tokens"] <= max_seq_len)
    documents: DocInfo[:] = doc_arr
    pool = DocumentPool(max_seq_len, pool_size)
    cdef int size = len(doc_arr)
    cdef int j
    for j in range(min(pool_size, size)):
        pool.add(documents[j].doc_id, documents[j].num_tokens)

    cdef cnp.ndarray out_arr = np.zeros(size, dtype=DOC_SEQUENCE_DTYPE)
    cdef DocSequence[:] out = out_arr

    cdef:
        long long out_ix = 0
        long long on_seq = 0
        long long on = 0
        long long seq_len = 0
        long long n_from_pool = 0
        long long stop_add_to_pool = len(doc_arr) - pool_size
        long long on_example = min(len(doc_arr), pool_size)
        long long new_tokens, next_len, remainder, in_pool, candidate_total
        long long best_on = 0
        long long best_pool = pool.find_at_most(max_seq_len)
        long long best_total = best_pool
    cdef cnp.ndarray in_progress_arr = np.zeros(max_seq_len, dtype=DOC_TOKENS_DTYPE)
    cdef DocTokens[:] in_progress = in_progress_arr

    while True:
        if on_example == size:
            if pool.is_empty():
                if on == 0:
                    break  # nothing left and nothing in-progress
                else:
                    new_tokens = max_seq_len + 1  # yield the in-progress sequence
            else:
                new_tokens = pool.get_first()  # Ran out of examples, start consuming the pool
        else:
            new_tokens = documents[on_example].num_tokens
            if new_tokens == max_seq_len:
                # Effectively 0 length since we will write them out separately
                new_tokens = 0
        next_len = seq_len + new_tokens

        if next_len > max_seq_len:
            # Write the best candidate we have found so far
            for j in range(best_on):
                if in_progress[j].num_tokens == max_seq_len:
                    out[out_ix].doc_id = in_progress[j].doc_id
                    out[out_ix].sequence_number = on_seq
                    out_ix += 1
                    on_seq += 1

            for j in range(best_on):
                if in_progress[j].num_tokens != max_seq_len:
                    out[out_ix].doc_id = in_progress[j].doc_id
                    out[out_ix].sequence_number = on_seq
                    out_ix += 1

            remainder = on - best_on  # num in-progress examples we are discarding
            for j in range(max(on-n_from_pool, best_on), on):
                # If our in-progress sequence used examples from the pool, move the unused example back to
                # the pool. We need this since the best sequence might have used one of these examples
                pool.add(in_progress[j].doc_id, in_progress[j].num_tokens)
                remainder -= 1
            on_example -= remainder

            if best_pool > 0:
                out[out_ix].doc_id = pool.pop(best_pool)
                out[out_ix].sequence_number = on_seq
                out_ix += 1
                if on_example < stop_add_to_pool:
                    pool.add(documents[on_example].doc_id, documents[on_example].num_tokens)  # re-fill the pool
                    on_example += 1

            on_seq += 1
            best_pool = pool.find_at_most(max_seq_len)
            best_total = best_pool
            on, on, seq_len, n_from_pool = 0, 0, 0, 0
        else:
            seq_len = next_len
            if on_example == size:
                n_from_pool += 1
                in_progress[on].doc_id = pool.pop(new_tokens)
                in_progress[on].num_tokens = new_tokens
            else:
                in_progress[on].doc_id = documents[on_example].doc_id
                in_progress[on].num_tokens = documents[on_example].num_tokens
                on_example += 1
            on += 1
            # Build a new candidate using the optimal sequence in the pool
            remainder = max_seq_len - seq_len
            in_pool = pool.find_at_most(remainder)
            candidate_total = seq_len + in_pool
            if candidate_total > best_total:
                best_on = on
                best_total = candidate_total
                best_pool = in_pool
    return out_arr