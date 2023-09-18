import math
from logging import getLogger

import numpy as np
import torch
import torch.nn as nn
import transformers

from ..triton_utils.mixin import TritonModuleMixin

logger = getLogger(__name__)

try:
    from ..triton_utils.kernels import (
        QuantLinearFunction,
        QuantLinearInferenceOnlyFunction,
        quant_matmul_248,
        quant_matmul_inference_only_248,
        transpose_quant_matmul_248,
    )
except ImportError:
    logger.error("triton not installed.")
    raise


class QuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable=False):
        super().__init__()
        if bits not in [2, 4, 8]:
            raise NotImplementedError("Only 2,4,8 bits are supported.")
        if infeatures % 32 != 0 or outfeatures % 32 != 0:
            raise NotImplementedError("in_feature and out_feature must be divisible by 32.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2**self.bits - 1

        self.register_buffer(
            "qweight", torch.zeros((infeatures // 32 * self.bits, outfeatures), dtype=torch.int32)
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (math.ceil(infeatures / self.group_size), outfeatures // 32 * self.bits), dtype=torch.int32
            ),
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(infeatures / self.group_size), outfeatures), dtype=torch.float16)
        )
        self.register_buffer(
            "g_idx", torch.tensor([i // self.group_size for i in range(infeatures)], dtype=torch.int32)
        )
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float16))
        else:
            self.bias = None

        self.trainable = trainable

    def post_init(self):
        pass

    def pack(self, linear, scales, zeros, g_idx=None):
        W = linear.weight.data.clone()
        if isinstance(linear, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(linear, transformers.pytorch_utils.Conv1D):
            W = W.t()

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx

        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        scale_zeros = zeros * scales
        self.scales = scales.clone().half()
        if linear.bias is not None:
            self.bias = linear.bias.clone().half()

        intweight = []
        for idx in range(self.infeatures):
            intweight.append(
                torch.round((W[:, idx] + scale_zeros[self.g_idx[idx]]) / self.scales[self.g_idx[idx]]).to(
                    torch.int
                )[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((intweight.shape[0] // 32 * self.bits, intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qweight[row] |= intweight[j] << (self.bits * (j - i))
                i += 32 // self.bits
                row += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

        zeros -= 1
        zeros = zeros.numpy().astype(np.uint32)
        qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 32 * self.bits), dtype=np.uint32)
        i = 0
        col = 0
        while col < qzeros.shape[1]:
            if self.bits in [2, 4, 8]:
                for j in range(i, i + (32 // self.bits)):
                    qzeros[:, col] |= zeros[:, j] << (self.bits * (j - i))
                i += 32 // self.bits
                col += 1
            else:
                raise NotImplementedError("Only 2,4,8 bits are supported.")

        qzeros = qzeros.astype(np.int32)
        self.qzeros = torch.from_numpy(qzeros)

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.outfeatures,)
        quant_linear_fn = QuantLinearFunction if self.trainable else QuantLinearInferenceOnlyFunction
        out = quant_linear_fn.apply(
            x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, self.g_idx, self.bits, self.maxq
        )
        out = out.half().reshape(out_shape)
        out = out + self.bias if self.bias is not None else out
        return out

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        """
        Pre-tunes the quantized kernel
        """
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.outfeatures

            if (k, n) not in kn_values:
                kn_values[(k, n)] = (m.qweight, m.scales, m.qzeros, m.g_idx, m.bits, m.maxq)

        logger.info(f"Found {len(kn_values)} unique KN Linear values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (qweight, scales, qzeros, g_idx, bits, maxq) in kn_values.items():
                    if transpose:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                        a = torch.randn(m, n, dtype=torch.float16, device=model.device)
                        transpose_quant_matmul_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
                    else:
                        a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                        quant_matmul_inference_only_248(a, qweight, scales, qzeros, g_idx, bits, maxq)
        del kn_values


__all__ = ["QuantLinear"]
