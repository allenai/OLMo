from typing import Optional

import torch
import triton

from olmo.triton.layer_norm import layer_norm


@triton.testing.perf_report(
    [
        triton.testing.Benchmark(
            x_names=["N"],
            x_vals=[512 * i for i in range(2, 32)],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["Triton", "Torch"],
            styles=[("blue", "-"), ("green", "-"), ("orange", "-")],
            ylabel="GB/s",
            plot_name=f"layer-norm-{'affine' if elementwise_affine else 'no-affine'}-{mode}",
            args={
                "M": 4096,
                "dtype": torch.float16,
                "mode": mode,
                "elementwise_affine": elementwise_affine,
            },
        )
        for mode in ("forward", "backward")
        for elementwise_affine in (True, False)
    ]
)
def bench_layer_norm(
    M,
    N,
    dtype,
    provider,
    mode="backward",
    elementwise_affine: bool = True,
    eps=1e-5,
    device="cuda",
):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight: Optional[torch.Tensor] = None
    bias: Optional[torch.Tensor] = None
    if elementwise_affine:
        weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
        bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == "triton":

        def y_fwd():
            return layer_norm(x, w_shape, weight=weight, bias=bias, eps=eps)

    elif provider == "torch":

        def y_fwd():
            return torch.nn.functional.layer_norm(x, w_shape, weight=weight, bias=bias, eps=eps)

    else:
        raise NotImplementedError(provider)

    # forward pass
    if mode == "forward":
        gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6  # type: ignore
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    # backward pass
    elif mode == "backward":

        def gbps(ms):
            return 3 * x.numel() * x.element_size() / ms * 1e-6

        y = y_fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: y.backward(dy, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            rep=500,
        )
    else:
        raise NotImplementedError(mode)

    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    print("Running benchmarks...")
    bench_layer_norm.run(print_data=True, show_plots=False)
