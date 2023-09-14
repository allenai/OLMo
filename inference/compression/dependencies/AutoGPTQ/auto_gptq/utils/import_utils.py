from logging import getLogger

import torch
from packaging.version import parse as parse_version

try:
    import triton

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

try:
    import autogptq_cuda_64
    import autogptq_cuda_256

    AUTOGPTQ_CUDA_AVAILABLE = True
except:
    AUTOGPTQ_CUDA_AVAILABLE = False


try:
    import exllama_kernels

    EXLLAMA_KERNELS_AVAILABLE = True
except:
    EXLLAMA_KERNELS_AVAILABLE = False

logger = getLogger(__name__)


def dynamically_import_QuantLinear(
    use_triton: bool, desc_act: bool, group_size: int, bits: int, disable_exllama: bool = False
):
    if use_triton:
        if torch.version.hip:
            logger.warning(
                "Running GPTQ triton version on AMD GPUs is untested and may result in errors or wrong predictions. Please use use_triton=False."
            )

        from ..nn_modules.qlinear.qlinear_triton import QuantLinear
    else:
        if bits == 4 and not disable_exllama and EXLLAMA_KERNELS_AVAILABLE:
            from ..nn_modules.qlinear.qlinear_exllama import QuantLinear
        elif not desc_act or group_size == -1:
            from ..nn_modules.qlinear.qlinear_cuda_old import QuantLinear
        else:
            from ..nn_modules.qlinear.qlinear_cuda import QuantLinear

    return QuantLinear


def compare_transformers_version(version: str = "v4.28.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from transformers import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))


def compare_pytorch_version(version: str = "v2.0.0", op: str = "eq"):
    assert op in ["eq", "lt", "le", "gt", "ge"]

    from torch import __version__

    return getattr(parse_version(__version__), f"__{op}__")(parse_version(version))
