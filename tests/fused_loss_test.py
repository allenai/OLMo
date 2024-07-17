import pytest
import torch
from torch.testing import assert_close

from olmo.train import cross_entropy_loss, fused_loss_fn


@pytest.mark.skipif(torch.cuda.device_count() < 1, reason="Requires CUDA device")
@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", (16, 64))
@pytest.mark.parametrize("seq_len", (57, 300))
@pytest.mark.parametrize("vocab_size", (100, 200))
@pytest.mark.parametrize("z_loss_multiplier", (1e-4, 1e-5))
def test_fused_loss(batch_size, seq_len, vocab_size, z_loss_multiplier):
    logits = torch.randn(batch_size * seq_len, vocab_size).cuda()
    labels = torch.randint(0, vocab_size, (batch_size * seq_len,)).cuda()

    loss, z_loss = cross_entropy_loss(logits, labels, compute_z_loss=True, z_loss_multiplier=z_loss_multiplier)
    f_loss, f_z_loss = fused_loss_fn(logits, labels, compute_z_loss=True, z_loss_multiplier=z_loss_multiplier)

    # Note: This is allowing for very big differences!
    assert_close(loss, f_loss, atol=1e-2, rtol=1e-3)
    assert_close(z_loss, f_z_loss, atol=1e-2, rtol=1e-3)
