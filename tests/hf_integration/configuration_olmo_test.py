from hf_integration.configuration_olmo import OLMoConfig
from olmo.config import ModelConfig


def test_config_save(model_path: str):
    config = ModelConfig(alibi=True)  # default is False
    hf_config = OLMoConfig(**config.asdict())

    hf_config.save_pretrained(model_path)
    loaded_hf_config = OLMoConfig.from_pretrained(model_path)

    assert hf_config.to_dict() == loaded_hf_config.to_dict()

    for key, val in config.asdict().items():
        assert getattr(loaded_hf_config, key) == val
