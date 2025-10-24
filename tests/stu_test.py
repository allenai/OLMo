"""Tests for STU (Spectral Transform Unit) integration."""

import pytest
import torch

from olmo.config import ModelConfig
from olmo.model import OLMo
from olmo.stu import get_hankel, get_spectral_filters, nearest_power_of_two


class TestSTUUtils:
    """Test STU utility functions."""

    def test_nearest_power_of_two_round_up(self):
        assert nearest_power_of_two(1, round_up=True) == 1
        assert nearest_power_of_two(2, round_up=True) == 2
        assert nearest_power_of_two(3, round_up=True) == 4
        assert nearest_power_of_two(7, round_up=True) == 8
        assert nearest_power_of_two(1024, round_up=True) == 1024

    def test_nearest_power_of_two_round_down(self):
        assert nearest_power_of_two(1, round_up=False) == 1
        assert nearest_power_of_two(2, round_up=False) == 2
        assert nearest_power_of_two(3, round_up=False) == 2
        assert nearest_power_of_two(7, round_up=False) == 4
        assert nearest_power_of_two(1024, round_up=False) == 1024

    def test_get_hankel_shape(self):
        seq_len = 128
        hankel = get_hankel(seq_len, use_hankel_L=False)
        assert hankel.shape == (seq_len, seq_len)

    def test_get_hankel_L_shape(self):
        seq_len = 128
        hankel_L = get_hankel(seq_len, use_hankel_L=True)
        assert hankel_L.shape == (seq_len, seq_len)

    def test_get_spectral_filters_shape(self):
        seq_len = 128
        num_eigh = 24
        phi = get_spectral_filters(seq_len, num_eigh, use_hankel_L=False)
        assert phi.shape == (seq_len, num_eigh)

    def test_get_spectral_filters_dtype(self):
        seq_len = 128
        num_eigh = 24
        phi = get_spectral_filters(seq_len, num_eigh, dtype=torch.float32)
        assert phi.dtype == torch.float32


class TestSTUModel:
    """Test STU model integration."""

    @pytest.fixture
    def base_config(self):
        """Create a minimal config for testing."""
        return ModelConfig(
            d_model=256,
            n_heads=4,
            n_layers=4,
            max_sequence_length=512,
            vocab_size=1000,
            embedding_size=1024,
            init_device="cpu",
            include_bias=False,
        )

    def test_model_without_stu(self, base_config):
        """Test that model works without STU layers."""
        model = OLMo(base_config, init_params=True)
        assert model is not None
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, base_config.embedding_size)

    def test_model_with_all_stu(self, base_config):
        """Test model with all STU layers."""
        base_config.stu_layer_schedule = "all"
        base_config.stu_num_eigh = 16  # Smaller for faster tests
        
        model = OLMo(base_config, init_params=True)
        assert model is not None
        
        # Verify STU blocks were created
        from olmo.stu import OLMoSTUBlock
        if base_config.block_group_size == 1:
            for block in model.transformer.blocks:
                assert isinstance(block, OLMoSTUBlock), f"Expected OLMoSTUBlock, got {type(block)}"
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, base_config.embedding_size)

    def test_model_with_alternating_stu(self, base_config):
        """Test model with alternating STU and attention layers."""
        base_config.stu_layer_schedule = "alternating"
        base_config.stu_num_eigh = 16
        
        model = OLMo(base_config, init_params=True)
        assert model is not None
        
        # Verify alternating pattern
        from olmo.stu import OLMoSTUBlock
        from olmo.model import OLMoSequentialBlock
        
        if base_config.block_group_size == 1:
            for i, block in enumerate(model.transformer.blocks):
                if i % 2 == 0:
                    assert isinstance(block, OLMoSTUBlock), f"Layer {i} should be STU"
                else:
                    assert isinstance(block, OLMoSequentialBlock), f"Layer {i} should be attention"
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, base_config.embedding_size)

    def test_model_with_attention_last(self, base_config):
        """Test model with STU everywhere except the last layer."""
        base_config.stu_layer_schedule = "attention_last"
        base_config.stu_num_eigh = 16
        
        model = OLMo(base_config, init_params=True)
        assert model is not None
        
        # Verify pattern: STU for all but last
        from olmo.stu import OLMoSTUBlock
        from olmo.model import OLMoSequentialBlock
        
        if base_config.block_group_size == 1:
            for i, block in enumerate(model.transformer.blocks):
                if i < base_config.n_layers - 1:
                    assert isinstance(block, OLMoSTUBlock), f"Layer {i} should be STU"
                else:
                    assert isinstance(block, OLMoSequentialBlock), f"Layer {i} should be attention"
        
        # Test forward pass
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, base_config.vocab_size, (batch_size, seq_len))
        
        output = model(input_ids)
        assert output.logits.shape == (batch_size, seq_len, base_config.embedding_size)

    def test_stu_config_parameters(self, base_config):
        """Test that STU config parameters are properly used."""
        base_config.stu_layer_schedule = "all"
        base_config.stu_num_eigh = 32
        base_config.stu_use_hankel_L = True
        base_config.stu_use_approx = False
        
        model = OLMo(base_config, init_params=True)
        
        # Check that the first STU block has the right config
        from olmo.stu import OLMoSTUBlock
        if base_config.block_group_size == 1:
            first_block = model.transformer.blocks[0]
            if isinstance(first_block, OLMoSTUBlock):
                assert first_block.stu.K == 32
                assert first_block.stu.use_hankel_L is True
                assert first_block.stu.use_approx is False

    def test_invalid_stu_schedule(self, base_config):
        """Test that invalid STU schedules raise errors."""
        base_config.stu_layer_schedule = "invalid_schedule"
        
        with pytest.raises(Exception):  # Should raise OLMoConfigurationError
            model = OLMo(base_config, init_params=True)
            # Try to call the method that would fail
            _ = model._should_use_stu_layer(0)

