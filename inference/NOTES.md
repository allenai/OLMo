# LLM inference

The goal here is to run inference for all OLMo models (up to 70 GB) on a single A100. Our approach for this at present is *post-hoc model quantization*.

Here's the [inference workstream google doc](https://docs.google.com/document/d/1DpCOsmTluGS0NDutgV7h_QNtiiVC8ocUEqEzqG76yfM/edit?usp=sharing).

## Available methods

We chatted with [Tim Dettmers](https://timdettmers.com/), who is an expert on model quantization.

- According to Tim, [GPTQ](https://arxiv.org/abs/2210.17323) is state-of-the-art for post-hoc 4-bit model quantization. Based on this, we're currently using GPTQ.
- Tim has more recently released [QLoRA](https://arxiv.org/abs/2305.14314), which can be used for 4-bit finetuning. I'm not sure if this technique is relevant for our use case. **It might be worth checking whether we should switch to this**, because the code is likely easier to work with (it's available through Huggingface).

## GPTQ implementations

There are a number of implementations available for GPTQ:

- Original GPTQ code from paper authors: <https://github.com/IST-DASLab/gptq>.
- GPTQ-for-LLaMa: <https://github.com/qwopqwop200/GPTQ-for-LLaMa>. What is sounds like; this is GPTQ adapted to work with LLaMa models.
- AutoGPTQ: <https://github.com/PanQiWei/AutoGPTQ>. This builds on the original code, but it's more nicely-engineered and makes it pretty easy to add new models via inheritance. **This is what we're using now**.

### Progress so far

#### Compressing LLaMa models with GPTQ

I've used AutoGPTQ to compress Hamish and Yizhong's instructed-tuned LLaMa models. Models up to 70B run on a single GPU. Code to do this is here: <https://github.com/allenai/open-instruct/tree/compress/quantize>. Most of my work is on the `compress` branch; some of it has been merged into `main` but not all.

There's also some [code](https://github.com/allenai/open-instruct/tree/compress/quantize/efficiency-benchmark) to run Hao's [efficiency benchmarking code](https://github.com/allenai/efficiency-benchmark) on compressed models. I haven't examined the results of this thoroughly, but the code runs and provides stats on energy usage, latency, etc.

##### Things that could be improved

- Inference latency. Roughly 200ms / token for the 70B model. It's possible that [hidet](https://pytorch.org/blog/introducing-hidet/) could speed this up. It's also possible that the AutoGPTQ code is just better now than it was a month ago and that latency would be lower if the models were quantized now.
- Evaluation. I've implemented accuracy evaluation MMLU (see [eval_on_mmlu](https://github.com/allenai/open-instruct/blob/compress/quantize/scripts/eval_on_mmlu.sh), but evals on more datasets would be good. This requires slightly modifying the evaluation code to accommodate `AutoGPTQ` models, as I did [here](https://github.com/allenai/open-instruct/blob/compress/eval/mmlu_eval/evaluate_hf_lm.py#LL114C18-L114C18).
  - I added the results to [Yizhong's spreadsheet](https://docs.google.com/spreadsheets/d/1jt_bkJXBmNN5ZmEFZg4NKsu8F9PtKpHWwG1WcqNb17E/edit?usp=sharing) under the tab `efficiency-stuff`; right now it's just a single number that confirms that nothing disastrous happens on MMLU at 65B. This could be improved.

#### Implementing GPTQ for OLMo

There are two steps:

- Add an `olmo.py` module [here](https://github.com/PanQiWei/AutoGPTQ/tree/main/auto_gptq/modeling) that inherits from `BaseGPTQForCausalLM`; see [llama.py](https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling/llama.py) for an example.
- Register the olmo implementation in the list of [supported models](https://github.com/PanQiWei/AutoGPTQ/blob/main/auto_gptq/modeling/_const.py#L10).

To add an `olmo.py` module, we can basically just imitate what was done for other models (e.g. LLaMa), mapping LLaMa model component names to OLMo names (Akshita, I shared a version of what I have for this). To do the mapping, I just instantiated a couple models and printed them out. I'm including a dump of all the models below for reference.

There's one important wrinkle here: some OLMo models use *fused linear attention*. I'm not sure how GPTQ handles this or whether any existing supported models implement attention the same way. This might be something to discuss with Dirk and Pete.

```python
OLMo(
  (transformer): ModuleDict(
    (wte): Embedding(50304, 768)
    (emb_drop): Dropout(p=0.1, inplace=False)
    (blocks): ModuleList(
      (0-11): 12 x OLMoSequentialBlock(
        (dropout): Dropout(p=0.1, inplace=False)
        (norm): LayerNorm()
        (act): SwiGLU()
        (attn_out): Linear(in_features=768, out_features=768, bias=True)
        (ff_out): Linear(in_features=1536, out_features=768, bias=True)
        (att_proj): Linear(in_features=768, out_features=2304, bias=True)
        (ff_proj): Linear(in_features=768, out_features=3072, bias=True)
      )
    )
    (ln_f): LayerNorm()
    (wpe): Embedding(1024, 768)
  )
)

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-3): 4 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)

BloomModel(
  (word_embeddings): Embedding(250880, 64)
  (word_embeddings_layernorm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
  (h): ModuleList(
    (0-1): 2 x BloomBlock(
      (input_layernorm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (self_attention): BloomAttention(
        (query_key_value): Linear(in_features=64, out_features=192, bias=True)
        (dense): Linear(in_features=64, out_features=64, bias=True)
        (attention_dropout): Dropout(p=0.0, inplace=False)
      )
      (post_attention_layernorm): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
      (mlp): BloomMLP(
        (dense_h_to_4h): Linear(in_features=64, out_features=256, bias=True)
        (gelu_impl): BloomGelu()
        (dense_4h_to_h): Linear(in_features=256, out_features=64, bias=True)
      )
    )
  )
  (ln_f): LayerNorm((64,), eps=1e-05, elementwise_affine=True)
)

GPTJForCausalLM(
  (transformer): GPTJModel(
    (wte): Embedding(50400, 4096)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-27): 28 x GPTJBlock(
        (ln_1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attn): GPTJAttention(
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (out_proj): Linear(in_features=4096, out_features=4096, bias=False)
        )
        (mlp): GPTJMLP(
          (fc_in): Linear(in_features=4096, out_features=16384, bias=True)
          (fc_out): Linear(in_features=16384, out_features=4096, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=50400, bias=True)
)
```
