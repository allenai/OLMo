from inspect import signature
from typing import Dict, Optional, Union

from ._base import BaseGPTQForCausalLM, BaseQuantizeConfig
from ._utils import check_and_get_model_type
from .baichuan import BaiChuanGPTQForCausalLM
from .bloom import BloomGPTQForCausalLM
from .codegen import CodeGenGPTQForCausalLM
from .gpt2 import GPT2GPTQForCausalLM
from .gpt_bigcode import GPTBigCodeGPTQForCausalLM
from .gpt_neox import GPTNeoXGPTQForCausalLM
from .gptj import GPTJGPTQForCausalLM
from .internlm import InternLMGPTQForCausalLM
from .llama import LlamaGPTQForCausalLM
from .moss import MOSSGPTQForCausalLM
from .olmo import OLMoGPTQForCausalLM
from .opt import OPTGPTQForCausalLM
from .qwen import QwenGPTQForCausalLM
from .rw import RWGPTQForCausalLM

GPTQ_CAUSAL_LM_MODEL_MAP = {
    "bloom": BloomGPTQForCausalLM,
    "gpt_neox": GPTNeoXGPTQForCausalLM,
    "gptj": GPTJGPTQForCausalLM,
    "gpt2": GPT2GPTQForCausalLM,
    "llama": LlamaGPTQForCausalLM,
    "olmo": OLMoGPTQForCausalLM,
    "opt": OPTGPTQForCausalLM,
    "moss": MOSSGPTQForCausalLM,
    "gpt_bigcode": GPTBigCodeGPTQForCausalLM,
    "codegen": CodeGenGPTQForCausalLM,
    "RefinedWebModel": RWGPTQForCausalLM,
    "RefinedWeb": RWGPTQForCausalLM,
    "baichuan": BaiChuanGPTQForCausalLM,
    "internlm": InternLMGPTQForCausalLM,
    "qwen": QwenGPTQForCausalLM,
}


class AutoGPTQForCausalLM:
    def __init__(self):
        raise EnvironmentError(
            "AutoGPTQModelForCausalLM is designed to be instantiated\n"
            "using `AutoGPTQModelForCausalLM.from_pretrained` if want to quantize a pretrained model.\n"
            "using `AutoGPTQModelForCausalLM.from_quantized` if want to inference with quantized model."
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        quantize_config: BaseQuantizeConfig,
        max_memory: Optional[dict] = None,
        trust_remote_code: bool = False,
        **model_init_kwargs
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(pretrained_model_name_or_path, trust_remote_code)
        return GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            quantize_config=quantize_config,
            max_memory=max_memory,
            trust_remote_code=trust_remote_code,
            **model_init_kwargs
        )

    @classmethod
    def from_quantized(
        cls,
        model_name_or_path: Optional[str],
        device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = None,
        max_memory: Optional[dict] = None,
        device: Optional[Union[str, int]] = None,
        low_cpu_mem_usage: bool = False,
        use_triton: bool = False,
        inject_fused_attention: bool = True,
        inject_fused_mlp: bool = True,
        use_cuda_fp16: bool = True,
        quantize_config: Optional[BaseQuantizeConfig] = None,
        model_basename: Optional[str] = None,
        use_safetensors: bool = False,
        trust_remote_code: bool = False,
        warmup_triton: bool = False,
        trainable: bool = False,
        disable_exllama: bool = False,
        **kwargs
    ) -> BaseGPTQForCausalLM:
        model_type = check_and_get_model_type(model_name_or_path, trust_remote_code)
        quant_func = GPTQ_CAUSAL_LM_MODEL_MAP[model_type].from_quantized
        # A static list of kwargs needed for huggingface_hub
        huggingface_kwargs = [
            "cache_dir",
            "force_download",
            "proxies",
            "resume_download",
            "local_files_only",
            "use_auth_token",
            "revision",
            "subfolder",
            "_raise_exceptions_for_missing_entries",
            "_commit_hash",
        ]
        # TODO: do we need this filtering of kwargs? @PanQiWei is there a reason we can't just pass all kwargs?
        keywords = {
            key: kwargs[key]
            for key in list(signature(quant_func).parameters.keys()) + huggingface_kwargs
            if key in kwargs
        }
        return quant_func(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            max_memory=max_memory,
            device=device,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_triton=use_triton,
            inject_fused_attention=inject_fused_attention,
            inject_fused_mlp=inject_fused_mlp,
            use_cuda_fp16=use_cuda_fp16,
            quantize_config=quantize_config,
            model_basename=model_basename,
            use_safetensors=use_safetensors,
            trust_remote_code=trust_remote_code,
            warmup_triton=warmup_triton,
            trainable=trainable,
            disable_exllama=disable_exllama,
            **keywords
        )


__all__ = ["AutoGPTQForCausalLM"]
