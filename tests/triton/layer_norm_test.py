from typing import Optional

import pytest
import torch

try:
    import triton as _  # noqa: F401

    has_triton = True
except ModuleNotFoundError:
    has_triton = False


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.skipif(not has_triton, reason="Requires triton")
@pytest.mark.parametrize("M, N", [(1151, 8192)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
def test_layer_norm_with_affine(M, N, dtype, eps=1e-5, device="cuda"):
    from olmo.triton.layer_norm import layer_norm  # type: ignore

    torch.manual_seed(23412467)
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [t.grad.clone() for t in [x, weight, bias]]  # type: ignore
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [t.grad.clone() for t in [x, weight, bias]]  # type: ignore
    # compare
    atol: Optional[float] = None
    rtol: Optional[float] = None
    if dtype == torch.bfloat16:
        atol = 7e-2
        rtol = 0.0
    elif dtype == torch.float16:
        atol = 1e-2
        rtol = 0.0
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx_tri, dx_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(db_tri, db_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dw_tri, dw_ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.skipif(not has_triton, reason="Requires triton")
@pytest.mark.parametrize("M, N", [(1151, 8192)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
def test_layer_norm_with_linear(M, N, dtype, eps=1e-5, device="cuda"):
    from olmo.triton.layer_norm import layer_norm  # type: ignore

    torch.manual_seed(23412467)
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, w_shape, weight=weight, eps=eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight=weight, eps=eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = [t.grad.clone() for t in [x, weight]]  # type: ignore
    x.grad, weight.grad = None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [t.grad.clone() for t in [x, weight]]  # type: ignore
    # compare
    atol: Optional[float] = None
    rtol: Optional[float] = None
    if dtype == torch.bfloat16:
        atol = 7e-2
        rtol = 0.0
    elif dtype == torch.float16:
        atol = 1e-2
        rtol = 0.0
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx_tri, dx_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dw_tri, dw_ref, atol=atol, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.skipif(not has_triton, reason="Requires triton")
@pytest.mark.parametrize("M, N", [(1151, 8192)])
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.float16, id="fp16"),
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float32, id="fp32"),
    ],
)
def test_layer_norm_no_affine(M, N, dtype, eps=1e-5, device="cuda"):
    from olmo.triton.layer_norm import layer_norm  # type: ignore

    torch.manual_seed(23423)
    # create data
    x_shape = (M, N)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, (N,), eps=eps)
    y_ref = torch.nn.functional.layer_norm(x, (N,), eps=eps).to(dtype)
    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    assert x.grad is not None
    dx_tri = x.grad.clone()
    x.grad = None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    assert x.grad is not None
    dx_ref = x.grad.clone()
    # compare
    atol: Optional[float] = None
    rtol: Optional[float] = None
    if dtype == torch.bfloat16:
        atol = 2e-2
        rtol = 0.0
    elif dtype == torch.float16:
        atol = 1e-2
        rtol = 0.0
    torch.testing.assert_close(y_tri, y_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(dx_tri, dx_ref, atol=atol, rtol=rtol)
