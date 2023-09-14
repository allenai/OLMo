from argparse import ArgumentParser

import datasets
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.eval_tasks import LanguageModelingTask
from transformers import AutoTokenizer

DATASET = "tatsu-lab/alpaca"
WITH_INPUT_TEMPLATE = "Instruction:\n{instruction}\n\nInput:\n{input}\n\nOutput:\n"
WITHOUT_INPUT_TEMPLATE = "Instruction:\n{instruction}\n\nOutput:\n"


def ds_refactor_fn(samples):
    instruction_data = samples["instruction"]
    input_data = samples["input"]
    output_data = samples["output"]

    new_samples = {"prompt": [], "output": []}
    for instruction_txt, input_txt, output_txt in zip(instruction_data, input_data, output_data):
        if input_txt:
            prompt = WITH_INPUT_TEMPLATE.format(instruction=instruction_txt, input=input_txt)
        else:
            prompt = WITHOUT_INPUT_TEMPLATE.format(instruction=instruction_txt)
        new_samples["prompt"].append(prompt)
        new_samples["output"].append(output_txt)

    return new_samples


def main():
    parser = ArgumentParser()
    parser.add_argument("--base_model_dir", type=str)
    parser.add_argument("--quantized_model_dir", type=str)
    parser.add_argument(
        "--num_samples", type=int, default=100, help="how many samples will be sampled to evaluation"
    )
    parser.add_argument("--sample_max_len", type=int, default=1024, help="max tokens for each sample")
    parser.add_argument("--block_max_len", type=int, default=2048, help="max tokens for each data block")
    parser.add_argument("--use_triton", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir)

    model = AutoGPTQForCausalLM.from_pretrained(args.base_model_dir, BaseQuantizeConfig())
    model.to("cuda:0")

    task = LanguageModelingTask(
        model=model,
        tokenizer=tokenizer,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="output",
        **{
            "num_samples": args.num_samples,  # how many samples will be sampled to evaluation
            "sample_max_len": args.sample_max_len,  # max tokens for each sample
            "block_max_len": args.block_max_len,  # max tokens for each data block
            "load_fn": datasets.load_dataset,  # function to load dataset
            "preprocess_fn": ds_refactor_fn,  # function to preprocess dataset
            "truncate_prompt": False,  # truncate label when sample's length exceed sample_max_len
        },
    )

    print(f"eval result for base model: {task.run()}")
    task.model = None
    model.cpu()
    del model
    torch.cuda.empty_cache()

    model = AutoGPTQForCausalLM.from_quantized(
        args.quantized_model_dir, device="cuda:0", use_triton=args.use_triton
    )
    task.model = model
    task.device = model.device
    print(f"eval result for quantized model: {task.run()}")


if __name__ == "__main__":
    main()
