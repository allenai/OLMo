<div align="center">
  <br>
  <br>
  <h1>OLMo: 开源语言模型 (Open Language Model)</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://github.com/allenai/OLMo/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/OLMo.svg">
  </a>
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
  <a href="https://playground.allenai.org">
    <img alt="Playground" src="https://img.shields.io/badge/Ai2-Playground-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# ⚠️ 注意事项 ⚠️ 本代码库对于最新版本已归档，不再处于活跃维护状态。有关最新的 OLMo 模型发布与前沿更新，请访问全新训练框架：https://github.com/allenai/OLMo-core/ 

OLMo 是用于训练和使用 AI2 顶尖开源语言模型的代码库。它由科学家精心设计，专为科学研究而生。

## 安装指南

首先，根据您的操作系统指引安装 [PyTorch](https://pytorch.org)。

对于模型预训练与微调，我们推荐从源码安装：

```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]
```

也可以通过 PyPI 直接安装：

```bash
pip install ai2-olmo
```

## 预训练 (Pretraining)

OLMo 的预训练遵循两阶段训练流程：
在第一阶段，我们在海量以网页为主的语料上进行训练：[OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)；
在第二阶段，我们在体量较小、高质量且针对性强的合成与精选数据上进行退火训练：[Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)。

您可以在 OLMo core 与 Hugging Face 格式中找到*所有*中间检查点（至少每 1000 步记录一次）：

| 规格变体 | OLMo 格式 (第一阶段) | OLMo 格式 (第二阶段) | Hugging Face 格式 |
|:---|:---|:---|:---|
| **OLMo-2 1B** | [OLMo-2 1B](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo-2-0425-1B.csv) | [OLMo-2 1B](https://github.com/allenai/OLMo/blob/main/configs/official-0425/OLMo-2-0425-1B-stage2.csv) | [1B 规格的 Hugging Face 权重](https://huggingface.co/allenai/OLMo-2-0425-1B) |
| **OLMo-2 7B** | [OLMo-2 7B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-7B.csv) | [OLMo-2 7B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-7B-stage2.csv) | [7B 规格的 Hugging Face 权重](https://huggingface.co/allenai/OLMo-2-1124-7B) |
| **OLMo-2 13B** | [OLMo-2 13B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-13B.csv) | [OLMo-2 13B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-13B-stage2.csv) | [13B 规格的 Hugging Face 权重](https://huggingface.co/allenai/OLMo-2-1124-13B) |
| **OLMo-2 32B** | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2-0325-32B.csv) | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo-2-0325-32B-stage2.csv) | [32B 规格的 Hugging Face 权重](https://huggingface.co/allenai/OLMo-2-0325-32B) |

> 注：32B 变体基于我们全新的训练器训练。若要训练或微调 OLMo-2 32B，请访问 [OLMo-core](https://github.com/allenai/OLMo-core)。

### 实验复现步骤

若要复现下文描述的任意训练流程，请执行以下命令：

```bash
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config}
```

针对训练配置，可选用下列列出的任何配置文件。

如果希望直接覆盖训练配置中的某些参数而无需每次都新建配置文件，可以这样操作：

```bash
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
  --setting1=value \
  --setting2=value \
  --setting3.subsetting1=value
```

下文给出的训练配置预设为通过 HTTP 在线流式加载训练数据。为了进行大规模复现，建议将数据文件下载至本地并修改路径指向本地文件系统。

#### 在 Apple Silicon (Mac) 设备上运行：

```bash
python scripts/train.py {path_to_train_config}
```

示例：

```bash
python scripts/train.py configs/tiny/OLMo-20M.yaml --save_overwrite
```

> 注：需升级 PyTorch 至 2.5.x 方可正常运行。

### 第一阶段 (Stage 1)

第一阶段是核心主训练阶段，我们在主要是网页抓取的大规模语料上训练 4 万亿（4T）或 5 万亿（5T）Tokens。

| | OLMo2 1B | OLMo2 7B | OLMo2 13B |
|:---|:---|:---|:---|
| **训练 Token 总量** | 4 万亿 (4 Trillion) | 4 万亿 (4 Trillion) | 5 万亿 (5 Trillion) |
| **模型检查点** | [stage1-step1907359-tokens4001B](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/stage1-step1907359-tokens4001B) | [stage1-step928646-tokens3896B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage1-step928646-tokens3896B) | [stage1-step596057-tokens5001B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage1-step596057-tokens5001B) |
| **训练配置** | [OLMo2-1B-stage1.yaml](configs/official-0425/OLMo2-1B-stage1.yaml) | [OLMo2-7B-stage1.yaml](configs/official-1124/OLMo2-7B-stage1.yaml) | [OLMo2-13B-stage1.yaml](configs/official-1124/OLMo2-13B-stage1.yaml) |
| **WandB 实验记录** | [wandb.ai/OLMo2-1B](https://api.wandb.ai/links/ai2-llm/izdtrtu0) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/OLMo-2-7B-Nov-2024--VmlldzoxMDUzMzE1OA) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |

包含训练数据出处追溯的 .csv.gz 文件可见[此处配置](configs/official-1124/provenance.csv)。

### 1B 模型第二阶段 (Stage 2)

针对 1B 模型，我们在 500 亿（50B）高质量 Tokens 上以不同数据顺序进行了 3 次重复训练，并将随机种子 42 的最终检查点作为最终基座模型：

| | 模型检查点 | 训练配置 | WandB 实验记录 |
|:---|:---|:---|:---|
| **随机种子 42069** | [stage2-ingredient1-step23852-tokens51B](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/stage2-ingredient1-step23852-tokens51B) | [OLMo2-1B-stage2-seed42069.yaml](configs/official-0425/OLMo2-1B-stage2-seed42069.yaml) | [wandb.ai/OLMo2-1B](https://api.wandb.ai/links/ai2-llm/izdtrtu0) |
| **随机种子 666** | [stage2-ingredient2-step23852-tokens51B](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/stage2-ingredient2-step23852-tokens51B) | [OLMo2-1B-stage2-seed666.yaml](configs/official-0425/OLMo2-1B-stage2-seed666.yaml) | [wandb.ai/OLMo2-1B](https://api.wandb.ai/links/ai2-llm/izdtrtu0) |
| **随机种子 42 (主模型)** | [stage2-ingredient3-step23852-tokens51B](https://huggingface.co/allenai/OLMo-2-0425-1B/tree/stage2-ingredient3-step23852-tokens51B) | [OLMo2-1B-stage2-seed42.yaml](configs/official-0425/OLMo2-1B-stage2-seed42.yaml) | [wandb.ai/OLMo2-1B](https://api.wandb.ai/links/ai2-llm/izdtrtu0) |

### 7B 模型第二阶段 (Stage 2)

针对 7B 模型，我们在 500 亿（50B）高质量 Tokens 上以不同数据顺序训练 3 次，随后对模型权重进行模型权重平均融合（Model Souping）：

| | 模型检查点 | 训练配置 | WandB 实验记录 |
|:---|:---|:---|:---|
| **随机种子 42** | [stage2-ingredient1-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient1-step11931-tokens50B) | [OLMo2-7B-stage2-seed42.yaml](configs/official-1124/OLMo2-7B-stage2-seed42.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| **随机种子 42069** | [stage2-ingredient2-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient2-step11931-tokens50B) | [OLMo2-7B-stage2-seed42069.yaml](configs/official-1124/OLMo2-7B-stage2-seed42069.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| **随机种子 666** | [stage2-ingredient3-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient3-step11931-tokens50B) | [OLMo2-7B-stage2-seed666.yaml](configs/official-1124/OLMo2-7B-stage2-seed666.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| **最终 Souped 融合模型** | [main](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/main) | 无配置，直接在 Python 中对检查点权重进行线性平均 | |

此处链接的训练配置预设为自动下载第一阶段的最新检查点，并以此为基础启动后续训练。

### 13B 模型第二阶段 (Stage 2)

针对 13B 模型，我们在 1000 亿（100B）高质量 Tokens 上以不同数据顺序训练 3 次，并在 3000 亿（300B）高质量 Tokens 上额外训练 1 次，随后对模型权重进行平均融合（Model Souping）：

| | 模型检查点 | 训练配置 | WandB 实验记录 |
|:---|:---|:---|:---|
| **随机种子 1110 (100B)** | [stage2-ingredient1-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient1-step11931-tokens100B) | [OLMo2-13B-stage2-seed1110-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed1110-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| **随机种子 2662 (100B)** | [stage2-ingredient2-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient2-step11931-tokens100B) | [OLMo2-13B-stage2-seed2662-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| **随机种子 6209 (100B)** | [stage2-ingredient3-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11931-tokens100B) | [OLMo2-13B-stage2-seed6209-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed6209-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| **随机种子 2662 (300B)** | [stage2-ingredient4-step11931-tokens300B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient4-step35773-tokens300B) | [OLMo2-13B-stage2-seed2662-300B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-300B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| **最终 Souped 融合模型** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main) | 无配置，直接在 Python 中对检查点权重进行平均 | |

> 注：有关 32B 模型的完整信息请参阅 [OLMo-core](https://github.com/allenai/OLMo-core) 仓库。

## 指令微调变体 (Instruct Variants)

有关各模型的官方指令微调版本，请访问：
 * [OLMo2 1B Instruct](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)
 * [OLMo2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
 * [OLMo2 13B Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)
 * [OLMo2 32B Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)

## 模型推理 (Inference)

您可以通过 Hugging Face 集成直接加载 OLMo 检查点并执行推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# 可选：转移至 CUDA 运行
# inputs = {k: v.to('cuda') for k, v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

也可以通过 Hugging Face pipeline 高层抽象直接调用：

```python
from transformers import pipeline

olmo_pipe = pipeline("text-generation", model="allenai/OLMo-2-0425-1B")
print(olmo_pipe("Language modeling is"))
```

### 模型量化 (Quantization)

```python
olmo = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMo-2-0425-1B",
    torch_dtype=torch.float16,
    load_in_8bit=True
)  # 需要安装 bitsandbytes
```

量化后的模型对输入数据类型与 CUDA 显存处理较为敏感。为避免潜在问题，建议显式将输入 ID 转移至 GPU：`inputs.input_ids.to('cuda')`。

## 模型评测 (Evaluation)

用于评估 OLMo 模型的完整评测工具链托管在 [OLMo Eval](https://github.com/allenai/OLMo-eval) 与 [olmes](https://github.com/allenai/olmes) 仓库。

## Modal.com 云端托管

在 `./scripts/olmo2_modal_openai.py` 中提供了一个通过 OpenAI 兼容 API 在 Modal.com 上托管 OLMo 2 模型的示例脚本。
运行步骤：

1. 遵循 [Modal.com 指南](https://modal.com/docs/guide) 中的入门指南安装 Modal 库和 CLI 命令行工具；
2. 遵循 Modal.com 指南中的 [Secrets](https://modal.com/docs/guide/secrets) 说明，创建一个名为 "example-secret-token" 的密钥，并定义环境变量 `MODAL_TOKEN`；
3. 执行部署命令：
```bash
modal deploy ./scripts/olmo2_modal_openai.py
```

部署完成后，可以使用类似以下的 curl 命令测试端点服务：

```bash
curl -X POST \
  -H "Authorization: Bearer [上述密钥令牌]" \
  -H "Content-Type: application/json" \
  -d @body.json \
  https://[Modal 分配的 Web 端点地址]/v1/chat/completions
```

其中 `body.json` 格式如下：

```json
{
    "model": "OLMo-2-1124-13B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Who was Alan Turing?"
        }
    ],
    "max_tokens": 100,
    "temperature": 0.9,
    "stream": true
}
```

## 论文引用

```bibtex
@misc{olmo20242olmo2furious,
      title={2 OLMo 2 Furious}, 
      author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656}, 
}
```

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
