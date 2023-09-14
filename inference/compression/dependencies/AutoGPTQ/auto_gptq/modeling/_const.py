from packaging.version import parse as parse_version
from torch import device

from ..utils.import_utils import compare_transformers_version

CPU = device("cpu")
CUDA_0 = device("cuda:0")

SUPPORTED_MODELS = [
    "bloom",
    "gptj",
    "gpt2",
    "gpt_neox",
    "opt",
    "olmo",
    "moss",
    "gpt_bigcode",
    "codegen",
    "RefinedWebModel",
    "RefinedWeb",
    "baichuan",
    "internlm",
    "qwen",
]
if compare_transformers_version("v4.28.0", op="ge"):
    SUPPORTED_MODELS.append("llama")

__all__ = ["CPU", "CUDA_0", "SUPPORTED_MODELS"]
