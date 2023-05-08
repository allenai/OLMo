from transformers import AutoTokenizer
import multiprocessing
import gzip
import json
import os
import argparse
import base64
import random
from tqdm import tqdm
import re
from transformers.utils import logging
logging.set_verbosity(40)

def serialize_int_list_to_bytes(int_list):
    # Assuming 2 bytes per integer with big-endian byte order
    byte_data = b''.join([i.to_bytes(2, 'big') for i in int_list])
    byte_data = base64.b64encode(byte_data).decode('utf-8')
    return byte_data

def deserialize_bytes_to_int_list(byte_data):
    byte_data = base64.b64decode(byte_data.encode('utf-8'))
    int_size = 2  # Assuming 2 bytes per integer
    int_list = [int.from_bytes(byte_data[i:i+int_size], 'big') for i in range(0, len(byte_data), int_size)]
    return int_list

def process_file(file, args):
    docs = []
    if args.subdomain_from_filename_regex is not None:
        match = re.match(args.subdomain_from_filename_regex, os.path.basename(file))
        if match is None:
            raise ValueError(f'File {file} does not match the subdomain_from_filename_regex')
        subdomain = match.group(1)
    with gzip.open(file, 'r') as fin:
        for line in fin:
            doc = json.loads(line)
            if args.subdomain_from_filename_regex is not None:
                doc['subdomain'] = subdomain
            if args.subdomain_from_metadata is not None:
                doc['subdomain'] = doc['metadata'][args.subdomain_from_metadata]
            docs.append(doc)
    return docs

def process_doc(doc, tokenizer):
    tokens = tokenizer(
        doc['text'],
        truncation=False,
        return_attention_mask=False
    )['input_ids']
    doc['tokens'] = tokens
    doc['tokenizer'] = tokenizer.name_or_path
    doc['token_length'] = len(tokens)

    return doc

def prepare_batch(batch_buffer, pad_token_id):
    max_len = max(len(doc['tokens']) for doc in batch_buffer)
    batch = []
    for doc in batch_buffer:
        batch.append({
            'id': doc['id'],
            'tokens': doc['tokens'] + [pad_token_id] * (max_len - len(doc['tokens'])),
            'token_len': len(doc['tokens']),
            'pad_tokens': max_len - len(doc['tokens'])
        })
    return batch, sum(doc['token_len'] for doc in batch), sum(doc['pad_tokens'] for doc in batch)

def process_sample(data, batch_size, max_tokenized_len, pad_token_id):
    token_count = 0
    pad_count = 0
    docs_output = []
    docs_output_buffer = []
    processed_batches = []
    processed_batch_buffer = []
    tokens_count_buffered = 0
    pad_count_buffered = 0
    batch_buffer = []

    
    for doc in tqdm(data, desc='Processing sampled documents'):
        docs_output_buffer.append(doc)
        tokens = doc['tokens']
        while len(tokens) > 0:
            batch_buffer.append({'id': doc['id'], 'tokens': tokens[:max_tokenized_len]})
            tokens = tokens[max_tokenized_len:]
            if len(batch_buffer) == batch_size:
                processed_batch, new_tokens_count, new_pad_count = prepare_batch(batch_buffer, pad_token_id)
                batch_buffer = []
                processed_batch_buffer.extend(processed_batch)
                tokens_count_buffered += new_tokens_count 
                pad_count_buffered += new_pad_count
        
        # flush the buffer only when a full batch is ready
        # docs left over after the last full batch will be dropped
        if len(processed_batch_buffer) > 0:
            docs_output.extend(docs_output_buffer)
            docs_output_buffer = []
            processed_batches.append(processed_batch_buffer)
            processed_batch_buffer = []
            token_count += tokens_count_buffered
            tokens_count_buffered = 0
            pad_count += pad_count_buffered
            pad_count_buffered = 0

    return docs_output, processed_batches, token_count, pad_count

def tokenize_data(args):
    # use os to set TOKENIZERS_PARALLELISM env variable to True to enable parallel tokenization
    os.environ['TOKENIZERS_PARALLELISM'] = 'False'

    data_files = sorted(args.data_files)
    file_args = [(file, args) for file in data_files]
   
    with multiprocessing.Pool() as pool:
        docs_per_file = pool.starmap(process_file, tqdm(file_args, total=len(file_args), desc="unziping data"))
        print("done unzipping data")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    assert tokenizer.vocab_size < 2**16
    num_docs = sum(len(docs) for docs in tqdm(docs_per_file, desc="counting docs"))
    def generate_doc_args(docs_per_file):
        for docs in docs_per_file:
            for doc in docs:
                yield (doc, tokenizer)
    with multiprocessing.Pool() as pool:
        tokenized_docs = pool.starmap(process_doc, tqdm(generate_doc_args(docs_per_file), total=num_docs, desc="tokenizing data"))
        print("done tokenizing data")

    return tokenized_docs

def load_file(file):
    output = []
    with gzip.open(file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc['tokens'] = deserialize_bytes_to_int_list(doc['tokens'])
            output.append(doc)
        return output

def load_pretokenized_data(args):
    data_files = sorted(args.pretokenized_data_files)

    with multiprocessing.Pool() as pool:
        data_per_file = list(tqdm(pool.imap(load_file, data_files), total=len(data_files)))
        data = [doc for file in data_per_file for doc in file]

    return data

def main(args):
    
    if args.data_files is not None:
        full_data = tokenize_data(args)
    else:
        full_data = load_pretokenized_data(args)


    subset_stats = {}
    for token_quota in args.token_quotas:
        splits = {}
        if token_quota is not None:
            datasets_token_len = sum(doc['token_length'] for doc in full_data)
            sample_ratio = token_quota / datasets_token_len
            print(f'Sampling {sample_ratio * 100:.2f}% of the documents')
            if args.test_and_val:
                sample_ratio *= 2
                assert sample_ratio <= 1, 'Sample ratio must be <= 0.5 when test and val are both used'
                sample = random.sample(full_data, int(len(full_data) * sample_ratio))
                splits['test'] = sample[:len(sample) // 2]
                splits['val'] = sample[len(sample) // 2:]
            else:
                splits[None] = random.sample(full_data, int(len(full_data) * sample_ratio))
        else:
            if args.test_and_val:
                sample_ratio = 1.0
                sample = random.sample(full_data, int(len(full_data) * sample_ratio))
                splits['test'] = sample[:len(sample) // 2]
                splits['val'] = sample[len(sample) // 2:]
            else:
                splits[None] = full_data

        for name, data in splits.items():
            token_quota_name = 'full data' if token_quota is None else f'{token_quota // 1000000}M Token Subset {name if name is not None else ""}'
            subset_stats[token_quota_name] = {}
            data.sort(key=lambda x: x['token_length'], reverse=True)

            for batch_size in args.batch_sizes:
                batch_size_name = 'full data' if batch_size is None else f'{batch_size} Batch Size'
                subset_stats[token_quota_name][batch_size_name] = {}
                docs_output, processed_batches, token_count, pad_count = process_sample(
                    data, batch_size, args.max_tokenized_len, args.pad_token_id
                )

                print(f'Sample has {len(docs_output)} documents with {token_count} tokens and {pad_count} pad tokens')
                subset_stats[token_quota_name][batch_size_name]['docs'] = len(docs_output)
                subset_stats[token_quota_name][batch_size_name]['tokens'] = token_count
                subset_stats[token_quota_name][batch_size_name]['pads'] = pad_count
                print(f'wasted comute is {pad_count / (token_count + pad_count) * 100:.2%}')
                subset_stats[token_quota_name][batch_size_name][r'wasted comute %'] = pad_count / (token_count + pad_count) * 100
                print(f'number of batches is {len(processed_batches)}')
                subset_stats[token_quota_name][batch_size_name]['batches'] = len(processed_batches)

                if args.stats_output_file is not None:
                    with open(args.stats_output_file, 'w') as f:
                        json.dump(subset_stats, f, indent=4)



                if args.output_dir is not None:
                    docs_output_serialized = []
                    for doc in docs_output:
                        doc['tokens'] = serialize_int_list_to_bytes(doc['tokens'])
                        docs_output_serialized.append(doc)

                    # write the data to file shards of 1000 documents each
                    for i in range(0, len(docs_output_serialized), 1000):
                        shard = docs_output_serialized[i:i+1000]
                        # pad shard number with zeros to 8 digits
                        file_name = f'{args.file_prefix}_{i//1000:08d}.jsonl.gz'
                        split_dir = name if name is not None else ''
                        with gzip.open(os.path.join(args.output_dir, split_dir, file_name), 'wt') as f:
                            for doc in shard:
                                f.write(json.dumps(doc) + '\n')

                if args.preprocessed_output_dir is not None:
                    processed_docs_output = []
                    for batch in processed_batches:
                        for doc in batch:
                            doc['tokens'] = serialize_int_list_to_bytes(doc['tokens'])
                            processed_docs_output.append(doc)
                    # write the data to file shards of 1000 documents each
                    for i in range(0, len(processed_docs_output), 1000):
                        shard = processed_docs_output[i:i+1000]
                        # pad shard number with zeros to 8 digits
                        file_name = f'{args.file_prefix}_{i//1000:08d}.jsonl.gz'
                        split_dir = name if name is not None else ''
                        with gzip.open(os.path.join(args.preprocessed_output_dir, split_dir, file_name), 'wt') as f:
                            for doc in shard:
                                f.write(json.dumps(doc) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='gpt2')
    parser.add_argument('--token_quotas', type=int, nargs='+', default=[None])
    parser.add_argument('--test_and_val', action='store_true')
    parser.add_argument('--data_files', type=str, nargs='+')
    parser.add_argument('--pretokenized_data_files', type=str, nargs='+')
    parser.add_argument('--seed', type=int, default=519)
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[16])
    parser.add_argument('--max_tokenized_len', type=int, default=2048)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--preprocessed_output_dir', type=str)
    parser.add_argument('--pad_token_id', type=int, default=0)
    parser.add_argument('--stats_output_file', type=str)
    parser.add_argument('--file_prefix', type=str)
    parser.add_argument('--subdomain_from_filename_regex', type=str)
    parser.add_argument('--subdomain_from_metadata', type=str)
    args = parser.parse_args()

    assert args.data_files != args.pretokenized_data_files, 'Only one of data_files and pretokenized_data_files can be specified'

    main(args)