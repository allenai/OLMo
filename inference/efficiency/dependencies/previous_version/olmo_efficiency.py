# Code to run model using Hao's efficiency benchmark.
# To get this to run, do `pip install auto-gptq[triton]`

import argparse
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from hf_olmo import OLMoConfig, OLMoForCausalLM, OLMoTokenizerFast


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

    with open("inputs_file.txt", "w") as f:
        f.write(line)


class PredictWrapper:
    def __init__(self, pretrained_model_dir):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
        self.tokenizer.padding_size = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_dir
        )  # , device=device, use_triton=False)
        self.model = self.model.to(device)

    def predict(self, inputs):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
        ).input_ids
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(inputs, max_new_tokens=256)
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for output in outputs:
            yield output.strip()


class VLLMPredictWrapper:
    def __init__(self, pretrained_model_dir):
        self.model = LLM(pretrained_model_dir, gpu_memory_utilization=0.9, tensor_parallel_size=2)

    def predict(self, inputs):
        outputs = self.model.generate(inputs)
        for output in outputs:
            yield output.outputs[0].text.strip()


def get_args():
    parser = argparse.ArgumentParser(description="Run efficiency benchmark")
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to the unquantized model / Name of the unquantized huggingface model.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    predictor = PredictWrapper(args.pretrained_model)
    # predictor = VLLMPredictWrapper(args.pretrained_model)
    stdio_predictor_wrapper(predictor)
