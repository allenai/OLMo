import json
import os
from argparse import ArgumentParser
from functools import partial

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model
from auto_gptq.utils.data_utils import make_data_block, collate_data
from auto_gptq.utils.peft_utils import GPTQAdaLoraConfig
from peft import TaskType

parser = ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--sample_max_length", type=int, default=1024, help="max length of sample")
parser.add_argument("--block_max_length", type=int, default=1024, help="max length of data block(bunch of samples)")
parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
parser.add_argument("--use_fast_tokenizer", action="store_true")
args = parser.parse_args()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name_or_path = args.model_name_or_path
tokenizer_name_or_path = args.tokenizer_name_or_path or model_name_or_path

lr = args.lr
num_epochs = args.num_epochs

# creating model
peft_config = GPTQAdaLoraConfig(
    init_r=20,
    target_r=16,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=args.use_fast_tokenizer)
if not tokenizer.pad_token_id:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoGPTQForCausalLM.from_quantized(
    model_name_or_path,
    use_triton=True,
    warmup_triton=False,
    trainable=True,
    inject_fused_attention=True,
    inject_fused_mlp=False
)
model.warmup_triton()
device = model.device
model = get_gptq_peft_model(model, peft_config=peft_config, auto_find_all_linears=True, train_mode=True)
model.print_trainable_parameters()

# loading dataset
WITH_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n"
WITHOUT_INPUT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Output:\n"


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


ds = Dataset.from_generator(
    lambda: json.load(open("../quantization/dataset/alpaca_data_cleaned.json", "r", encoding="utf-8"))
)
ds = ds.map(
    make_data_block,
    batched=True,
    batch_size=len(ds),
    num_proc=1,
    remove_columns=ds.column_names,
    keep_in_memory=True,
    load_from_cache_file=False,
    fn_kwargs={
        "prompt_col_name": "prompt",
        "label_col_name": "output",
        "tokenizer": tokenizer,
        "preprocess_fn": ds_refactor_fn,
        "sample_max_len": args.sample_max_length,
        "block_max_len": args.block_max_length,
        "add_eos_token": True,
        "truncate_prompt": False,
        "merge_prompt_label": True
    }
)
ds = ds.train_test_split(test_size=len(ds) // 10)
train_ds, eval_ds = ds["train"], ds["test"]
collate_fn = partial(collate_data, pad_token_id=tokenizer.pad_token_id)
train_dataloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=partial(collate_fn))
eval_dataloader = DataLoader(eval_ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
model.base_model.peft_config["default"].total_step = len(train_dataloader) * num_epochs

# training and evaluation
with torch.cuda.amp.autocast():
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader)
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            # Update the importance of low-rank matrices
            # and allocate the budget accordingly.
            model.base_model.update_and_allocate(global_step)
            optimizer.zero_grad()
            global_step += 1

            progress_bar.set_postfix(loss=loss.item())

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

model.save_pretrained(os.path.join(model_name_or_path, f"gptq_{peft_config.peft_type.value}_adapter"))
