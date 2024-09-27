import argparse
import glob
import numpy as np
import os
import resource
import shutil
import sys
import time
from tqdm import tqdm

HACK = 100000

def tokenize(args):

    ds_path = os.path.join(args.save_dir, f'tokenized.0')
    od_path = os.path.join(args.save_dir, f'offset.0')
    if os.path.exists(ds_path) and os.path.exists(od_path):
        print('Step 1 (tokenize): Skipped. All tokenized files already exist.')
        return

    print('Step 1 (tokenize): Starting ...')
    start_time = time.time()

    eos_id = 50279

    data_paths = sum([list(sorted(glob.glob(data_path, recursive=True))) for data_path in args.data_paths], [])
    ds_fout = open(ds_path, 'wb')
    od_fout = open(od_path, 'wb')
    od = 0
    for data_path in tqdm(data_paths):
        ods = []
        with open(data_path, 'rb') as f:
            byte_arr = f.read()
        arr = np.frombuffer(byte_arr, dtype=np.uint16)
        if arr[-1] != eos_id:
            arr = np.concatenate((arr, np.array([eos_id], dtype=np.uint16)))
        arr = arr[::-1].copy()
        indices = np.where(arr == eos_id)[0]
        for i in indices:
            ods.append(od + 2 * i)
        arr[indices] = args.doc_sep_id
        byte_arr = arr.view(np.uint8).tobytes()
        ds_fout.write(byte_arr)
        od_fout.write(np.array(ods, dtype=np.uint64).view(np.uint8).tobytes())
        od += len(byte_arr)

    ds_fout.close()
    od_fout.close()

    end_time = time.time()
    print(f'Step 1 (tokenize): Done. Took {end_time-start_time:.2f} seconds')

def build_sa(args):

    ds_path = os.path.join(args.save_dir, f'tokenized.0')
    sa_path = os.path.join(args.save_dir, f'table.0')
    if os.path.exists(sa_path):
        print(f'Step 2 (build_sa): Skipped. Table already exists.')
        return

    print('Step 2 (build_sa): starting ...')
    start_time_all = time.time()

    # -------- Step 2.1 (make-part) -------- #

    print(f'Step 2.1 (make-part): starting ...')
    start_time = time.time()

    tok_size = os.path.getsize(ds_path)
    mem_bytes = args.mem * 1024**3
    num_job_batches = 1
    while num_job_batches * (mem_bytes // 8) < tok_size:
        num_job_batches *= 2
    parallel_jobs = args.cpus
    total_jobs = num_job_batches * parallel_jobs
    print(f'Using {num_job_batches} batches of {parallel_jobs} jobs each, for a total of {total_jobs} jobs.')

    S = tok_size // total_jobs
    # Make sure that parts contain whole tokens (2 bytes)
    if S % 2 == 1:
        S += 1

    parts_dir = os.path.join(args.temp_dir, f'parts')
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)

    ranges, files = [], []
    for batch_start in tqdm(list(range(0, total_jobs, parallel_jobs))):
        batch_end = min(batch_start+parallel_jobs, total_jobs)
        batch_ranges, batch_files = [], []
        for i in range(batch_start, batch_end):
            s, e = i*S, min((i+1)*S+HACK, tok_size)
            batch_ranges.append((s, e))
            batch_files.append(os.path.join(parts_dir, f'{s}-{e}'))
        ranges += batch_ranges
        files += batch_files
        wait = []
        for (s, e) in batch_ranges:
            cmd = f'./suffix_array make-part --data-file {ds_path} --parts-dir {parts_dir} --start-byte {s} --end-byte {e}'
            wait.append(os.popen(cmd))
        [x.read() for x in wait]

    end_time = time.time()
    print(f'Step 2.1 (make-part): done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.2 (merge) -------- #

    print(f'Step 2.2 (merge): starting ...')
    start_time = time.time()

    merged_dir = os.path.join(args.temp_dir, f'merged')
    shutil.rmtree(merged_dir, ignore_errors=True)
    os.makedirs(merged_dir)

    cmd = f'./suffix_array merge --merged-dir {merged_dir} --suffix-path {" --suffix-path ".join(files)} --num-threads {args.cpus} --hacksize {HACK}'
    pipe = os.popen(cmd)
    output = pipe.read()
    if pipe.close() is not None:
        print('Something went wrong with merging.')
        exit(1)

    shutil.rmtree(parts_dir)

    end_time = time.time()
    print(f'Step 2.2 (merge): done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.3 (concat) -------- #

    print(f'Step 2.3 (concat): starting ...')
    start_time = time.time()

    os.popen(f'cat {merged_dir}/* > {sa_path}').read()
    shutil.rmtree(merged_dir)

    end_time = time.time()
    print(f'Step 2.3 (concat): done. Took {end_time-start_time:.2f} seconds')

    # -------- Step 2.4 (verify) -------- #

    if not os.path.exists(sa_path):
        print('Failed to create table')
        exit(1)

    table_size = os.path.getsize(sa_path)
    if table_size % (tok_size // 2) != 0:
        print('File size is wrong')
        exit(1)

    end_time_all = time.time()
    print(f'Step 2 (build_sa): Done. Took {end_time_all-start_time_all:.2f} seconds')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_paths', type=str, nargs='+', required=True, help='Regex pattern(s) for matching data files. Must be absolute path.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--doc_sep_id', type=int, default=65535)
    parser.add_argument('--cpus', type=int, required=True, help='Typically should be a power of 2.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--ulimit', type=int, default=524288, help='Maximum number of open files allowed.')
    args = parser.parse_args()
    if args.temp_dir is None:
        args.temp_dir = args.save_dir
    args.temp_dir = args.temp_dir.rstrip('/')
    args.save_dir = args.save_dir.rstrip('/')

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    tokenize(args)
    build_sa(args)

if __name__ == '__main__':
    main()
