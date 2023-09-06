import pytest
import torch

from olmo.triton.layer_norm import layer_norm


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.parametrize("M, N, dtype", [(1151, 8192, torch.float16)])
def test_layer_norm_with_affine(M, N, dtype, eps=1e-5, device="cuda"):
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
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0)


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.parametrize("M, N, dtype", [(1151, 8192, torch.float16)])
def test_layer_norm_no_affine(M, N, dtype, eps=1e-5, device="cuda"):
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
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0)
