# Code to run model using Hao's efficiency benchmark.
# To get this to run, do `pip install auto-gptq[triton]`

import argparse
import json

import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from hf_olmo import *


def stdio_predictor_wrapper(predictor):
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


class ModelSetUp:
    def __init__(self, pretrained_model_dir, quantized_model_dir):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.tokenizer = OLMoTokenizerFast.from_pretrained(pretrained_model_dir, use_fast=False)
        self.tokenizer.padding_size = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if quantized_model_dir:
            self.model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device=device, use_triton=False)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_dir)
            self.model = self.model.to(device)

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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    predictor = ModelSetUp(args.pretrained_model_dir, args.quantized_model_dir)
    stdio_predictor_wrapper(predictor)
