from hf_olmo import *
import transformers

MODEL_NAME = '/net/nfs.cirrascale/allennlp/jiachengl/hb-wolf-olmo/ckpt_unsharded/v2.5_v2.4_shane-fix_step409000'
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, max_batch_size_per_device=1, local_rank=0, global_rank=0, local_world_size=1, world_size=1)
message = ["Barack Obama was born in"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = model.generate(**inputs, max_new_tokens=20, do_sample=False, use_cache=False)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
