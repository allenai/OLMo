"""
General script/tooling to:
1. Take in a set of .jsonl files (s3 or weka or whatever)
2. Load a model (huggingface or on weak or whatever)
3. Run inference on 'text' field of each json within the dataset and add the generated text to the 'output' field of the dataset

and let's just assume that I can run everything 
And have everything be runnable on a single node/single GPU
(and distribute in another way, using beaker-gantry experiments for each)
"""

import os
import json
import gzip
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError
import glob

# ========================================================
# =                   DATASET STUFF                      =
# ========================================================

class JSONLDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        self.data.extend(load_jsonl(file_path))



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_jsonl(input_path: str) -> list: 
    parsed_path = urlparse(input_path)

    if parsed_path.scheme == 's3':
        bucket_name = parsed_path.netloc
        object_key = parsed_path.path.lstrip('/')
        s3 = boto3.client('s3')
        try:
            response = s3.get_object(Bucket=bucket_name, Key=object_key)
            data = response['Body'].read()
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(f"The file does not exist: s3://{bucket_name}/{object_key}")
            else:
                raise
    else:
        # Local path
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The file does not exist: {input_path}")
        with open(input_path, 'rb') as f:
            data = f.read()
        print(f"File loaded locally: {input_path}")
    
    data = [json.loads(_) for _ in gzip.decompress(data).splitlines()]      
    return data


def save_jsonl(output_dicts, output_path):
    output_data = gzip.compress(b'\n'.join([json.dumps(_).encode('utf-8') for _ in output_dicts]))

    if output_path.startswith('s3://'):
        parsed_path = urlparse(output_path)
        bucket_name = parsed_path.netloc
        object_key = parsed_path.path.lstrip('/')
        
        s3 = boto3.client('s3')
        s3.put_object(Bucket=bucket_name, Key=object_key, Body=data)
        print(f"File saved to S3: s3://{bucket_name}/{object_key}")
    else:
        # Local path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(data)
        print(f"File saved locally: {output_path}")


def input_path_to_output_path(input_path: str, input_dir: str, output_dir: str) -> str:
    # Replaces input_dir in input_path with output_dir
    return input_path.replace(input_dir, output_dir)


def list_files(input_dir: str, part_num: int, num_parts: int, ext='.jsonl.gz') -> list:
    if input_dir.startswith('s3://'):
        parsed_uri = urlparse(input_dir)
        bucket_name = parsed_uri.netloc
        prefix = parsed_uri.path.lstrip('/')

        s3 = boto3.client('s3')
        matching_files = []

        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    filename = os.path.basename(key)
                    if filename.endswith(ext):
                        matching_files.append(f"s3://{bucket_name}/{key}")
    else:
        search_path = os.path.join(input_dir, '**', '*' + ext)
        matching_files = glob.glob(search_path, recursive=True)

    return [f for i, f in enumerate(matching_files) if i % num_parts == part_num]


def unroll_dict(dictlist: dict) -> list:
    ans = []
    keys = list(dictlist.keys())
    n = len(dictlist[keys[0]])
    for i in range(n):
        d = {}
        for k in keys:
            d[k] = dictlist[k][i]
        ans.append(d)
    return d




# ========================================================
# =                 MODEL LOADING STUFF                  =
# ========================================================

def load_llama(model_name):
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).cuda()
    return tokenizer, model

# ========================================================
# =                  INFERENCE CODE                      =
# ========================================================


def run_inference(args):
    # Load dataset -- do one dataset/loader per file to make it easy
    input_files = list_files(args.input_dir, args.part, args.num_parts)
    input_datasets = {k: JSONLDataset(k) for k in input_files}

    # Load model --
    if args.model_name == "meta-llama/Llama-3.1-8B":
        tokenizer, model = load_llama(args.model_name)
        model.eval()
    else:
        raise NotImplementedError()


    # And then do the inference
    with torch.no_grad():
        for f, dataset in input_datasets.items():
            results = []
            dataloader = DataLoader(dataset, batch_size=args.batch_size)

            for batch in tqdm(dataloader, desc="Processing %s" % f):
                input_text = batch['text']
                inputs = {k: v.cuda() for k, v in tokenizer(input_text, return_tensors="pt", padding=False, truncation=True, max_length=512).items()}
                outputs = model.generate(**inputs, max_length=args.max_length)
                decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                batch['output'] = decoded_outputs
                results.extend(unroll_dict(batch))

            output_path = input_path_to_output_path(f, args.input_dir, args.output_dir)
            save_jsonl(results, output_path)



# ========================================================
# =                      MAIN BLOCK                      =
# ========================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="One-off Inference for Language Models")
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.1-8B", help="Name of the model to use")
    parser.add_argument("--input-dir", type=str, required=True, help="Input of directory that contains .jsonl.gz files")
    parser.add_argument("--part", type=int, default=0, help="If we partition the input dir's files into many pieces, which part here")
    parser.add_argument("--num-parts", type=int, default=1, help="If we partition the input dir's files into one piece, how many parts ehre")
    parser.add_argument("--output-dir", type=str, required=True, help="Where the outputs should go")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum length of generated text")
   
    args = parser.parse_args()
    run_inference(args)







