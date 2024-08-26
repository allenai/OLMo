from hf_olmo.modeling_olmo import OLMoForCausalLM
from hf_olmo.tokenization_olmo_fast import OLMoTokenizerFast
from transformers import AutoTokenizer
import torch

model_dir = "local_checkpoints/peteish1-baseline/latest-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = OLMoForCausalLM.from_pretrained(model_dir).to(DEVICE)
tokenizer = OLMoTokenizerFast.from_pretrained(model_dir)
input_ids = tokenizer("Bitcoin is", return_tensors="pt")["input_ids"].to(DEVICE)
out = model.generate(input_ids, max_length=64)
print(tokenizer.decode(out[0]))