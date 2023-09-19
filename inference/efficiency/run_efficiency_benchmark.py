# Code to run model using Hao's efficiency benchmark.
# To get this to run, do `pip install auto-gptq[triton]`

import argparse
import json
import sys

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from hf_olmo import *  # noqa: F403,F401


class ModelWrapper:
    def __init__(self, pretrained_model_dir, quantized_model_dir):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        use_fast = "olmo" in pretrained_model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=use_fast)
        self.tokenizer.padding_size = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token or self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id or self.tokenizer.eos_token_id
        if quantized_model_dir:
            self.model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=device, use_triton=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir, device_map="auto")
            # self.model.to(device)

    def predict(self, inputs):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs=inputs, max_new_tokens=256)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output in outputs:
            yield output.strip()


class VLLMModel(ModelWrapper):
    def __init__(self, pretrained_model_dir: str):
        self.model = LLM(
            pretrained_model_dir, trust_remote_code=True, tensor_parallel_size=1
        )  # torch.cuda.device_count())

    def predict(self, inputs):
        sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
        outputs = self.model.generate(inputs, sampling_params=sampling_params)
        for output in outputs:
            yield output.outputs[0].text.strip()


def stdio_predictor_wrapper(predictor: ModelWrapper):
    """
    Wrap a predictor in a loop that reads from stdin and writes to stdout.
    The predictor implements `predict` function that takes a single string and returns the label.

    Assumes each input instance ends with "\n".
    """
    for line in sys.stdin:
        line = line.rstrip()
        inputs = json.loads(line)
        assert isinstance(inputs, list)
        # Participants need to connect their inference code to our wrapper through the following line.
        outputs = predictor.predict(inputs=inputs)
        # Writes are \n deliminated, so adding \n is essential to separate this write from the next loop iteration.
        outputs = [o for o in outputs]
        sys.stdout.write(f"{json.dumps(outputs)}\n")
        # Writes to stdout are buffered. The flush ensures the output is immediately sent through the pipe
        # instead of buffered.
        sys.stdout.flush()


def get_args():
    parser = argparse.ArgumentParser(description="Run efficiency benchmark")

    parser.add_argument(
        "--pretrained-model-dir",
        type=str,
        help="Path to the unquantized model / Name of the unquantized huggingface model.",
    )

    parser.add_argument(
        "--quantized-model-dir",
        type=str,
        default=None,
        nargs="?",
        help="Path to the quantized model / Name of the quantized huggingface model.",
    )

    parser.add_argument(
        "--vllm",
        action="store_true",
        default=False,
        help="Load model with vllm",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    predictor: ModelWrapper
    if args.vllm:
        predictor = VLLMModel(args.pretrained_model_dir)
    else:
        predictor = ModelWrapper(args.pretrained_model_dir, args.quantized_model_dir)
    stdio_predictor_wrapper(predictor)
