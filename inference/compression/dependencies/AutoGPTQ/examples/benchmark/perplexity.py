import argparse
import os

import torch
from auto_gptq.utils import Perplexity
from transformers import AutoTokenizer

if __name__ == "__main__":
    """
    Example usage.

    Default usage with GPT2 model:
    python examples/benchmark/perplexity.py

    Specify GPTQ quantized model:
    python examples/benchmark/perplexity.py \
        --model_name TheBloke/open-llama-7b-open-instruct-GPTQ \
        --model_basename gptq_model-4bit-128g \
        --is_quantized
    
    Change your dataset:
    python examples/benchmark/perplexity.py --dataset_path tiny_shakespeare

    """
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name.")
    parser.add_argument("--model_basename", type=str, default=None, help="Model file's basename.")
    parser.add_argument("--n_ctx", type=int, default=512, help="Context size.")
    parser.add_argument("--n_batch", type=int, default=512, help="Batch size.")
    parser.add_argument("--dataset_path", type=str, default="wikitext", help="Path to the dataset.")
    parser.add_argument("--dataset_name", type=str, default=None, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use.")
    parser.add_argument(
        "--text_column", type=str, default="text", help="Column in the dataset containing the text."
    )
    parser.add_argument("--per_gpu_max_memory", type=int, default=None, help="Max memory used in each GPU.")
    parser.add_argument("--cpu_max_memory", type=int, default=None, help="Mx memory used in CPU.")
    parser.add_argument("--is_quantized", action="store_true", help="Is the model GPTQ quantized?")
    parser.add_argument("--use_safetensors", action="store_true", help="Whether to use safetensors model file")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Wheter to use fast tokenizer")
    parser.add_argument("--trust_remote_code", action="store_true", help="Whether to use remote code")
    parser.add_argument("--disable_exllama", action="store_true", help="Whether to use disable exllama kernel")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=args.use_fast_tokenizer)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    max_memory = dict()
    if args.per_gpu_max_memory is not None and args.per_gpu_max_memory > 0:
        if torch.cuda.is_available():
            max_memory.update({i: f"{args.per_gpu_max_memory}GIB" for i in range(torch.cuda.device_count())})
    if args.cpu_max_memory is not None and args.cpu_max_memory > 0 and max_memory:
        max_memory["cpu"] = f"{args.cpu_max_memory}GIB"
    if not max_memory:
        max_memory = None

    if args.is_quantized:
        from auto_gptq import AutoGPTQForCausalLM

        model = AutoGPTQForCausalLM.from_quantized(
            args.model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            model_basename=args.model_basename,
            use_safetensors=args.use_safetensors,
            trust_remote_code=args.trust_remote_code,
            inject_fused_mlp=False,
            inject_fused_attention=False,
            disable_exllama=args.disable_exllama,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=args.trust_remote_code,
        )

    ppl = Perplexity(model, tokenizer, args.dataset_path, args.dataset_name, args.split, args.text_column)
    ppl.calculate_perplexity(args.n_ctx, args.n_batch)
