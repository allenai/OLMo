<h1 align="center">AutoGPTQ</h1>
<p align="center">一个基于 GPTQ 算法，简单易用且拥有用户友好型接口的大语言模型量化工具包。</p>
<p align="center">
    <a href="https://github.com/PanQiWei/AutoGPTQ/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/PanQiWei/AutoGPTQ.svg">
    </a>
    <a href="https://pypi.org/project/auto-gptq/">
        <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dd/auto-gptq">
    </a>
</p>
<h4 align="center">
    <p>
        <a href="https://github.com/PanQiWei/AutoGPTQ/blob/main/README.md">English</a> |
        <b>中文</b>
    </p>
</h4>

*<center>📣 好久不见！👋 七月和八月将会迎来架构升级，性能优化和新特性，敬请关注！🥂</center>*

## 新闻或更新

- 2023-08-23 - (新闻) - 🤗 Transformers、optimum 和 peft 完成了对 `auto-gptq` 的集成，现在使用 GPTQ 模型进行推理和训练将变得更容易！阅读 [这篇博客](https://huggingface.co/blog/gptq-integration) 和相关资源以了解更多细节！
- 2023-08-21 - (新闻) - 通义千问团队发布了基于 `auto-gptq` 的 Qwen-7B 4bit 量化版本模型，并提供了[详尽的测评结果](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4#%E9%87%8F%E5%8C%96-quantization)
- 2023-08-06 - (更新) - 支持 exllama 的 q4 CUDA 算子使得 int4 量化模型能够获得至少1.3倍的推理速度提升.
- 2023-08-04 - (更新) - 支持 RoCm 使得 AMD GPU 的用户能够使用 auto-gptq 的 CUDA 拓展.
- 2023-07-26 - (更新) - 一个优雅的 [PPL 测评脚本](examples/benchmark/perplexity.py)以获得可以与诸如 `llama.cpp` 等代码库进行公平比较的结果。
- 2023-06-05 - (更新) - 集成 🤗 peft 来使用 gptq 量化过的模型训练适应层，支持 LoRA，AdaLoRA，AdaptionPrompt 等。
- 2023-05-30 - (更新) - 支持从 🤗 Hub 下载量化好的模型或上次量化好的模型到 🤗 Hub。

*获取更多的历史信息，请转至[这里](docs/NEWS_OR_UPDATE.md)*

## 性能对比

### 推理速度
> 以下结果通过[这个脚本](examples/benchmark/generation_speed.py)生成，文本输入的 batch size 为1，解码策略为 beam search 并且强制模型生成512个 token，速度的计量单位为 tokens/s（越大越好）。
> 
> 量化模型通过能够最大化推理速度的方式加载。

| model         | GPU           | num_beams | fp16  | gptq-int4 |
|---------------|---------------|-----------|-------|-----------|
| llama-7b      | 1xA100-40G    | 1         | 18.87 | 25.53     |
| llama-7b      | 1xA100-40G    | 4         | 68.79 | 91.30     |
| moss-moon 16b | 1xA100-40G    | 1         | 12.48 | 15.25     |
| moss-moon 16b | 1xA100-40G    | 4         | OOM   | 42.67     |
| moss-moon 16b | 2xA100-40G    | 1         | 06.83 | 06.78     |
| moss-moon 16b | 2xA100-40G    | 4         | 13.10 | 10.80     |
| gpt-j 6b      | 1xRTX3060-12G | 1         | OOM   | 29.55     |
| gpt-j 6b      | 1xRTX3060-12G | 4         | OOM   | 47.36     |


### 困惑度（PPL）
对于困惑度的对比， 你可以参考 [这里](https://github.com/qwopqwop200/GPTQ-for-LLaMa#result) 和 [这里](https://github.com/qwopqwop200/GPTQ-for-LLaMa#gptq-vs-bitsandbytes)

## 安装

### 快速安装
你可以通过 pip 来安装与 PyTorch 2.0.1 相兼容的最新稳定版本的 AutoGPTQ 的预构建轮子文件：

* 对于 CUDA 11.7： `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu117/`
* 对于 CUDA 11.8： `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/`
* 对于 RoCm 5.4.2： `pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/rocm542/`

**警告：** 预构建的轮子文件不一定在 PyTorch 的 nightly 版本上有效。如果要使用 PyTorch 的 nightly 版本，请从源码安装 AutoGPTQ。

#### 取消 cuda 拓展的安装
默认情况下，在 `torch` 和 `cuda` 已经于你的机器上被安装时，cuda 拓展将被自动安装，如果你不想要这些拓展的话，采用以下安装命令：
```shell
BUILD_CUDA_EXT=0 pip install auto-gptq
```
同时为确保该拓展——`autogptq_cuda` 不再存在于你的虚拟环境，执行以下命令：
```shell
pip uninstall autogptq_cuda -y
```

#### 支持使用 triton 加速
若想使用 `triton` 加速模型推理，使用以下命令：
> 警告：目前 triton 仅支持 linux 操作系统；当使用 triton 时 3-bit 数值类型的量化将不被支持

```shell
pip install auto-gptq[triton]
```

### 从源码安装
<details>
<summary>点击以查看详情</summary>

克隆源码:
```shell
git clone https://github.com/PanQiWei/AutoGPTQ.git && cd AutoGPTQ
```
然后，从项目目录安装:
```shell
pip install .
```
正如在快速安装一节，你可以使用 `BUILD_CUDA_EXT=0` 来取消构建 cuda 拓展。

如果你想要使用 triton 加速且其能够被你的操作系统所支持，请使用 `.[triton]`。

对应 AMD GPUs，为了从源码安装以支持 RoCm，请设置 `ROCM_VERSION` 环境变量。同时通过设置 `PYTORCH_ROCM_ARCH` ([reference](https://github.com/pytorch/pytorch/blob/7b73b1e8a73a1777ebe8d2cd4487eb13da55b3ba/setup.py#L132)) 可提升编译速度，例如：对于 MI200 系列设备，该变量可设为 `gfx90a`。例子：

```
ROCM_VERSION=5.6 pip install .
```

对于 RoCm 系统，在从源码安装时额外需要提前安装以下包：`rocsparse-dev`, `hipsparse-dev`, `rocthrust-dev`, `rocblas-dev` and `hipblas-dev`。

</details>

## 快速开始

### 量化和推理
> 警告：这里仅是对 AutoGPTQ 中基本接口的用法展示，只使用了一条文本来量化一个特别小的模型，因此其结果的表现可能不如在大模型上执行量化后预期的那样好。

以下展示了使用 `auto_gptq` 进行量化和推理的最简单用法：
```python
from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit"


tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
examples = [
    tokenizer(
        "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]

quantize_config = BaseQuantizeConfig(
    bits=4,  # 将模型量化为 4-bit 数值类型
    group_size=128,  # 一般推荐将此参数的值设置为 128
    desc_act=False,  # 设为 False 可以显著提升推理速度，但是 ppl 可能会轻微地变差
)

# 加载未量化的模型，默认情况下，模型总是会被加载到 CPU 内存中
model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

# 量化模型, 样本的数据类型应该为 List[Dict]，其中字典的键有且仅有 input_ids 和 attention_mask
model.quantize(examples)

# 保存量化好的模型
model.save_quantized(quantized_model_dir)

# 使用 safetensors 保存量化好的模型
model.save_quantized(quantized_model_dir, use_safetensors=True)

# 将量化好的模型直接上传至 Hugging Face Hub 
# 当使用 use_auth_token=True 时, 确保你已经首先使用 huggingface-cli login 进行了登录
# 或者可以使用 use_auth_token="hf_xxxxxxx" 来显式地添加账户认证 token
# （取消下面三行代码的注释来使用该功能）
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

# 或者你也可以同时将量化好的模型保存到本地并上传至 Hugging Face Hub
# （取消下面三行代码的注释来使用该功能）
# repo_id = f"YourUserName/{quantized_model_dir}"
# commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
# model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

# 加载量化好的模型到能被识别到的第一块显卡中
model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

# 从 Hugging Face Hub 下载量化好的模型并加载到能被识别到的第一块显卡中
# model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

# 使用 model.generate 执行推理
print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

# 或者使用 TextGenerationPipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipeline("auto-gptq is")[0]["generated_text"])
```

参考 [此样例脚本](examples/quantization/quant_with_alpaca.py) 以了解进阶的用法。

### 自定义模型

<details>

<summary>以下展示了如何拓展 `auto_gptq` 以支持 `OPT` 模型，如你所见，这非常简单：</summary>

```python
from auto_gptq.modeling import BaseGPTQForCausalLM


class OPTGPTQForCausalLM(BaseGPTQForCausalLM):
    # chained attribute name of transformer layer block
    layers_block_name = "model.decoder.layers"
    # chained attribute names of other nn modules that in the same level as the transformer layer block
    outside_layer_modules = [
        "model.decoder.embed_tokens", "model.decoder.embed_positions", "model.decoder.project_out",
        "model.decoder.project_in", "model.decoder.final_layer_norm"
    ]
    # chained attribute names of linear layers in transformer layer module
    # normally, there are four sub lists, for each one the modules in it can be seen as one operation, 
    # and the order should be the order when they are truly executed, in this case (and usually in most cases), 
    # they are: attention q_k_v projection, attention output projection, MLP project input, MLP project output
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.out_proj"],
        ["fc1"],
        ["fc2"]
    ]
```
然后, 你就可以像在基本用法一节中展示的那样使用 `OPTGPTQForCausalLM.from_pretrained` 和其他方法。

</details>


### 在下游任务上执行评估
你可以使用在 `auto_gptq.eval_tasks` 中定义的任务来评估量化前后的模型在某个特定下游任务上的表现。

这些预定义的模型支持所有在 [🤗 transformers](https://github.com/huggingface/transformers)和本项目中被实现了的 causal-language-models。

<details>

<summary>以下是使用 `cardiffnlp/tweet_sentiment_multilingual` 数据集在序列分类（文本分类）任务上评估 `EleutherAI/gpt-j-6b` 模型的示例:</summary>

```python
from functools import partial

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from auto_gptq.eval_tasks import SequenceClassificationTask


MODEL = "EleutherAI/gpt-j-6b"
DATASET = "cardiffnlp/tweet_sentiment_multilingual"
TEMPLATE = "Question:What's the sentiment of the given text? Choices are {labels}.\nText: {text}\nAnswer:"
ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive"
}
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


#  model = AutoModelForCausalLM.from_pretrained(MODEL).eval().half().to("cuda:0")
model = AutoGPTQForCausalLM.from_pretrained(MODEL, BaseQuantizeConfig())
tokenizer = AutoTokenizer.from_pretrained(MODEL)

task = SequenceClassificationTask(
        model=model,
        tokenizer=tokenizer,
        classes=LABELS,
        data_name_or_path=DATASET,
        prompt_col_name="prompt",
        label_col_name="label",
        **{
            "num_samples": 1000,  # how many samples will be sampled to evaluation
            "sample_max_len": 1024,  # max tokens for each sample
            "block_max_len": 2048,  # max tokens for each data block
            # function to load dataset, one must only accept data_name_or_path as input 
            # and return datasets.Dataset
            "load_fn": partial(datasets.load_dataset, name="english"),  
            # function to preprocess dataset, which is used for datasets.Dataset.map, 
            # must return Dict[str, list] with only two keys: [prompt_col_name, label_col_name]
            "preprocess_fn": ds_refactor_fn,  
            # truncate label when sample's length exceed sample_max_len
            "truncate_prompt": False  
        }
    )

# note that max_new_tokens will be automatically specified internally based on given classes
print(task.run())

# self-consistency
print(
    task.run(
        generation_config=GenerationConfig(
            num_beams=3,
            num_return_sequences=3,
            do_sample=True
        )
    )
)
```

</details>

## 了解更多
[教程](docs/tutorial) 提供了将 `auto_gptq` 集成到你的项目中的手把手指导和最佳实践准则。

[示例](examples/README.md) 提供了大量示例脚本以将 `auto_gptq` 用于不同领域。

## 支持的模型

> 你可以使用 `model.config.model_type` 来对照下表以检查你正在使用的一个模型是否被 `auto_gptq` 所支持。
> 
> 比如， `WizardLM`，`vicuna` 和 `gpt4all` 模型的 `model_type` 皆为 `llama`， 因此这些模型皆被 `auto_gptq` 所支持。

| model type                         | quantization | inference | peft-lora | peft-ada-lora | peft-adaption_prompt                                                              |
|------------------------------------|--------------|-----------|-----------|---------------|-----------------------------------------------------------------------------------|
| bloom                              | ✅            | ✅         | ✅         | ✅             |                                                                                   |
| gpt2                               | ✅            | ✅         | ✅         | ✅             |                                                                                   |
| gpt_neox                           | ✅            | ✅         | ✅         | ✅             | ✅[要求该分支的 peft](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| gptj                               | ✅            | ✅         | ✅         | ✅             | ✅[要求该分支的 peft](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| llama                              | ✅            | ✅         | ✅         | ✅             | ✅                                                                                 |
| moss                               | ✅            | ✅         | ✅         | ✅             | ✅[要求该分支的 peft](https://github.com/PanQiWei/peft/tree/multi_modal_adaption_prompt) |
| opt                                | ✅            | ✅         | ✅         | ✅             |                                                                                   |
| gpt_bigcode                        | ✅            | ✅         | ✅         | ✅             |                                                                                   |
| codegen                            | ✅            | ✅         | ✅         | ✅             |                                                                                   |
| falcon(RefinedWebModel/RefinedWeb) | ✅            | ✅         | ✅         | ✅             |                                                                                   |

## 支持的评估任务
目前， `auto_gptq` 支持以下评估任务： `LanguageModelingTask`, `SequenceClassificationTask` 和 `TextSummarizationTask`；更多的评估任务即将到来！

## 致谢
- 特别感谢 **Elias Frantar**， **Saleh Ashkboos**， **Torsten Hoefler** 和 **Dan Alistarh** 提出 **GPTQ** 算法并开源[代码](https://github.com/IST-DASLab/gptq)。
- 特别感谢 **qwopqwop200**， 本项目中涉及到模型量化的代码主要参考自 [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda)。

[![Star History Chart](https://api.star-history.com/svg?repos=PanQiwei/AutoGPTQ&type=Date)](https://star-history.com/#PanQiWei/AutoGPTQ&Date)