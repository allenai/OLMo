import math
from logging import getLogger

import torch
from transformers.models.llama.modeling_llama import LlamaMLP

from ..utils.import_utils import TRITON_AVAILABLE
from ._fused_base import FusedBaseMLPModule

logger = getLogger(__name__)

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    from .triton_utils import custom_autotune
    from .triton_utils.kernels import silu

    @custom_autotune.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),  # 3090
            triton.Config(
                {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),  # 3090
            triton.Config(
                {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8},
                num_stages=2,
                num_warps=4,
            ),  # 3090
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 16, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),  # 3090
            triton.Config(
                {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8},
                num_stages=4,
                num_warps=4,
            ),  # 3090
        ],
        key=["M", "N", "K"],
        nearest_power_of_two=True,
        prune_configs_by={
            "early_config_prune": custom_autotune.matmul248_kernel_config_pruner,
            "perf_model": None,
            "top_k": None,
        },
    )
    @triton.jit
    def quant_fused_matmul_248_kernel(
        a_ptr,
        c_ptr,
        b1_ptr,
        scales1_ptr,
        zeros1_ptr,
        g1_ptr,
        b2_ptr,
        scales2_ptr,
        zeros2_ptr,
        g2_ptr,
        M,
        N,
        K,
        bits,
        maxq,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_scales,
        stride_zeros,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        """
        Computes: C = silu(A * B1) * (A * B2)
        A is of shape (M, K) float16
        B is of shape (K//8, N) int32
        C is of shape (M, N) float16
        scales is of shape (1, N) float16
        zeros is of shape (1, N//8) int32
        """
        infearure_per_bits = 32 // bits

        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        )  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        a_mask = offs_am[:, None] < M
        # b_ptrs is set up such that it repeats elements along the K axis 8 times
        b1_ptrs = b1_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
        b2_ptrs = b2_ptr + ((offs_k[:, None] // infearure_per_bits) * stride_bk + offs_bn[None, :] * stride_bn)
        g1_ptrs = g1_ptr + offs_k
        g2_ptrs = g2_ptr + offs_k
        # shifter is used to extract the N bits of each element in the 32-bit word from B
        scales1_ptrs = scales1_ptr + offs_bn[None, :]
        scales2_ptrs = scales2_ptr + offs_bn[None, :]
        zeros1_ptrs = zeros1_ptr + (offs_bn[None, :] // infearure_per_bits)
        zeros2_ptrs = zeros2_ptr + (offs_bn[None, :] // infearure_per_bits)

        shifter = (offs_k % infearure_per_bits) * bits
        zeros_shifter = (offs_bn % infearure_per_bits) * bits
        accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, num_pid_k):
            g1_idx = tl.load(g1_ptrs)
            g2_idx = tl.load(g2_ptrs)

            # Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
            scales1 = tl.load(scales1_ptrs + g1_idx[:, None] * stride_scales)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            scales2 = tl.load(scales2_ptrs + g2_idx[:, None] * stride_scales)

            zeros1 = tl.load(zeros1_ptrs + g1_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros1 = (zeros1 >> zeros_shifter[None, :]) & maxq
            zeros1 = zeros1 + 1

            zeros2 = tl.load(zeros2_ptrs + g2_idx[:, None] * stride_zeros)  # (BLOCK_SIZE_K, BLOCK_SIZE_N,)
            zeros2 = (zeros2 >> zeros_shifter[None, :]) & maxq
            zeros2 = zeros2 + 1

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)  # (BLOCK_SIZE_M, BLOCK_SIZE_K)
            b1 = tl.load(b1_ptrs)  # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
            b2 = tl.load(b2_ptrs)

            # Now we need to unpack b (which is N-bit values) into 32-bit values
            b1 = (b1 >> shifter[:, None]) & maxq  # Extract the N-bit values
            b1 = (b1 - zeros1) * scales1  # Scale and shift
            accumulator1 += tl.dot(a, b1)

            b2 = (b2 >> shifter[:, None]) & maxq
            b2 = (b2 - zeros2) * scales2
            accumulator2 += tl.dot(a, b2)

            a_ptrs += BLOCK_SIZE_K
            b1_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            b2_ptrs += (BLOCK_SIZE_K // infearure_per_bits) * stride_bk
            g1_ptrs += BLOCK_SIZE_K
            g2_ptrs += BLOCK_SIZE_K

        accumulator1 = silu(accumulator1)
        c = accumulator1 * accumulator2
        c = c.to(tl.float16)
        c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
        c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

else:
    quant_fused_matmul_248_kernel = None


class FusedLlamaMLPForQuantizedModel(FusedBaseMLPModule):
    def __init__(
        self,
        gate_proj,
        down_proj,
        up_proj,
    ):
        super().__init__()

        self.infeatures = gate_proj.infeatures
        self.intermediate_size = gate_proj.outfeatures
        self.outfeatures = down_proj.outfeatures
        self.bits = gate_proj.bits
        self.maxq = gate_proj.maxq

        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def forward(self, x):
        return self.down_proj(self.triton_llama_mlp(x))

    def triton_llama_mlp(self, x):
        with torch.cuda.device(x.device):
            out_shape = x.shape[:-1] + (self.intermediate_size,)
            x = x.reshape(-1, x.shape[-1])
            M, K = x.shape
            N = self.intermediate_size
            c = torch.empty((M, N), device=x.device, dtype=torch.float16)
            grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
            quant_fused_matmul_248_kernel[grid](
                x,
                c,
                self.gate_proj.qweight,
                self.gate_proj.scales,
                self.gate_proj.qzeros,
                self.gate_proj.g_idx,
                self.up_proj.qweight,
                self.up_proj.scales,
                self.up_proj.qzeros,
                self.up_proj.g_idx,
                M,
                N,
                K,
                self.bits,
                self.maxq,
                x.stride(0),
                x.stride(1),
                self.gate_proj.qweight.stride(0),
                self.gate_proj.qweight.stride(1),
                c.stride(0),
                c.stride(1),
                self.gate_proj.scales.stride(0),
                self.gate_proj.qzeros.stride(0),
            )
            c = c.reshape(out_shape)
            return c

    @classmethod
    def inject_to_model(cls, model, use_triton=False, **kwargs):
        if not use_triton:
            logger.warning(f"skip module injection for {cls.__name__} not support integrate without triton yet.")
            return
        elif not TRITON_AVAILABLE:
            logger.warning(f"skip module injection for triton is not installed.")
            return

        for name, m in model.named_modules():
            if not isinstance(m, LlamaMLP):
                continue

            mlp = cls(m.gate_proj, m.down_proj, m.up_proj)

            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, mlp)

    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        from tqdm import tqdm

        kn_values = {}

        for _, m in model.named_modules():
            if not isinstance(m, cls):
                continue

            k = m.infeatures
            n = m.intermediate_size

            if (k, n) not in kn_values:
                kn_values[(k, n)] = m

        logger.info(f"Found {len(kn_values)} unique fused mlp KN values.")
        logger.info("Warming up autotune cache ...")
        with torch.no_grad():
            for m in tqdm(range(0, math.ceil(math.log2(seqlen)) + 1)):
                m = 2**m
                for (k, n), (modules) in kn_values.items():
                    a = torch.randn(m, k, dtype=torch.float16, device=model.device)
                    modules.triton_llama_mlp(a)
        del kn_values


__all__ = ["FusedLlamaMLPForQuantizedModel"]
