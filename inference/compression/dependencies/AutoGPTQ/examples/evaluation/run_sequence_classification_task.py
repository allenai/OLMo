from argparse import ArgumentParser
from functools import partial

import datasets
import torch
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.eval_tasks import SequenceClassificationTask
from transformers import AutoTokenizer

DATASET = "cardiffnlp/tweet_sentiment_multilingual"
TEMPLATE = "Question:What's the sentiment of the given text? Choices are {labels}.\nText: {text}\nAnswer:"
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABELS = list(ID2LABEL.values())


def ds_refactor_fn(samples):
    text_data = samples["text"]
    label_data = samples["label"]

    new_samples = {"prompt": [], "label": []}
    for text, label in zip(text_data, label_data):
        prompt = TEMPLATE.format(labels=LABELS, text=text)
        new_samples["prompt"].append(prompt)
        new_samples["label"].append(ID2LABEL[label])

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

    task = SequenceClassificationTask(
        model=model,
        tokenizer=tokenizer,
        classes=LABELS,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="label",
        **{
            "num_samples": args.num_samples,  # how many samples will be sampled to evaluation
            "sample_max_len": args.sample_max_len,  # max tokens for each sample
            "block_max_len": args.block_max_len,  # max tokens for each data block
            "load_fn": partial(datasets.load_dataset, name="english"),  # function to load dataset
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
