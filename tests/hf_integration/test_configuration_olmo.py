from olmo.config import ModelConfig
from hf_integration.configuration_olmo import OLMoConfig


def test_config_save(path: str="test-path"):
    config = ModelConfig()
    hf_config = OLMoConfig(**config.asdict())

    hf_config.save_pretrained(path)
    loaded_hf_config = OLMoConfig.from_pretrained(path)

    assert hf_config.to_dict() == loaded_hf_config.to_dict()

    for key, val in config.asdict().items():
        assert getattr(loaded_hf_config, key) == val
