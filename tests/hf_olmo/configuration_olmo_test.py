from olmo.config import ModelConfig
import tempfile


def test_config_save(model_path: str):
    from hf_olmo.configuration_olmo import OLMoConfig

    with tempfile.TemporaryDirectory() as temp_dir:
        config = ModelConfig(alibi=True)  # default is False
        hf_config = OLMoConfig(**config.asdict())

        hf_config.save_pretrained(temp_dir)
        loaded_hf_config = OLMoConfig.from_pretrained(temp_dir)

        assert hf_config.to_dict() == loaded_hf_config.to_dict()

        for key, val in config.asdict().items():
            assert getattr(loaded_hf_config, key) == val
