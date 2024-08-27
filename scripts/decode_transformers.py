import transformers
MODEL_NAME = './ckpt_transformers/v2.5_v2.4_shane-fix_step409000'
# MODEL_NAME = './ckpt_transformers/v2.7_v2.5_vera_no-infgram_step11234'
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
message = ["Barack Obama was born in"]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)

logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits # (B, L, V)
logits = logits[0, -1] # (V,)
print(logits.topk(5))
exit()

response = model.generate(**inputs, max_new_tokens=10, do_sample=False, use_cache=False)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
