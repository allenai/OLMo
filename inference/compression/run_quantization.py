"""
Run 4-bit model quantization with GPTQ, using Wikitext as train data.
Based on `examples/quantization/basic_usage_wikitext2` in AutoGPT.

Usage example (runs on a single GPU):
python quantize_autogptq.py \
    --pretrained_model_dir "/net/nfs.cirrascale/allennlp/hamishi/open-instruct/alpaca_fixed_65b" \
    --quantized_model_dir "/net/nfs.cirrascale/allennlp/davidw/checkpoints/gptq_alpaca_fixed_65b"
"""


import argparse
import time

import numpy as np
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
from transformers import AutoTokenizer

from hf_olmo import *  # noqa: F403,F401


def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)

    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc


def get_args():
    parser = argparse.ArgumentParser(description="Run 4-bit model quantization using GPTQ.")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to the unquantized model / Name of the unquantized huggingface model.",
    )
    parser.add_argument("--quantized-model-dir", type=str, help="Output path for the quantized model.")
    parser.add_argument("--n-samples", type=int, help="Number of samples from Wikitext", default=128)
    args = parser.parse_args()

    return args


def main():
    "Run quantization."
    args = get_args()
    print("Getting data.")
    trainloader, testenc = get_wikitext2(args.n_samples, 0, 2048, args.pretrained_model)
    print("Done.")

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
    )

    print("Loading unquantized model")
    # Load un-quantized model, the model will always be force loaded into cpu
    model = AutoGPTQForCausalLM.from_pretrained(args.pretrained_model, quantize_config)
    print("Done")

    # Quantize model, the examples should be list of dict whose keys can only be
    # "input_ids" and "attention_mask" with value under torch.LongTensor type.
    print("Quantizing")
    tick = time.time()
    model.quantize(trainloader, use_triton=True)
    elapsed = (time.time() - tick) / 60
    print(f"Elapsed time:{elapsed:0.2f} minutes.")

    # save quantized model
    print("Saving")
    model.save_quantized(args.quantized_model_dir)
    print("Done")


if __name__ == "__main__":
    main()
