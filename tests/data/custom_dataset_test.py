from olmo.config import CustomDatasetConfig
from olmo.data.custom_datasets import build_custom_dataset, extract_module_and_class


def test_extract_module_and_class():
    assert extract_module_and_class("foo") == (None, "foo")
    assert extract_module_and_class("foo.bar") == ("foo", "bar")
    assert extract_module_and_class("foo.bar.Baz") == ("foo.bar", "Baz")
    assert extract_module_and_class("foo.bar.Baz.Quux") == ("foo.bar.Baz", "Quux")


def test_build_custom_dataset_full_path(train_config):
    import collections

    train_config.data.custom_dataset = CustomDatasetConfig(name="collections.Counter", args={"x": 9, "y": 4})
    dataset = build_custom_dataset(train_config)
    assert isinstance(dataset, collections.Counter)
    assert dataset["x"] == 9
    assert dataset["y"] == 4


def test_build_custom_dataset_module_specified(train_config):
    import collections

    train_config.data.custom_dataset = CustomDatasetConfig(
        name="Counter", module="collections", args={"x": 9, "y": 4}
    )
    dataset = build_custom_dataset(train_config)
    assert isinstance(dataset, collections.Counter)
    assert dataset["x"] == 9
    assert dataset["y"] == 4
