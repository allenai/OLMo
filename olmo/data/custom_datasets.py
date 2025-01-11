import importlib
import logging
from typing import Optional, Tuple

from torch.utils.data import Dataset

from ..config import TrainConfig
from ..exceptions import OLMoConfigurationError

__all__ = ["build_custom_dataset"]

LOGGER = logging.getLogger(__name__)


def build_custom_dataset(train_config: TrainConfig) -> Dataset:
    """Returns an instance custom dataset class from the specified custom module and class name in the train config.
    Assumes that the returned class implements the Pytorch Dataset interface. Responsibility
    for ensuring that the class implements the interface is left to the user.

    Instantiates the class with the arguments specified in the train config.

    :param train_config: OLMo train config
    :raises OLMoConfigurationError: if the custom dataset class is not specified or the module cannot be determined
    :return: Instantiated custom dataset class
    """
    assert train_config.data.custom_dataset is not None
    if not train_config.data.custom_dataset.name:
        raise OLMoConfigurationError("custom_dataset_class is required when using a custom dataset")
    LOGGER.warning("Using custom dataset class, deterministic training is not guaranteed")
    LOGGER.info(
        f"Loading custom dataset {train_config.data.custom_dataset.name} from module {train_config.data.custom_dataset.module}"
    )
    dataset_class = train_config.data.custom_dataset.name
    dataset_module = train_config.data.custom_dataset.module
    if not dataset_module:
        dataset_module, dataset_class = extract_module_and_class(dataset_class)
    if dataset_module is None:
        raise OLMoConfigurationError(
            "when using custom_dataset_class, use the full module path of the class or specify custom_dataset_module"
        )
    module = importlib.import_module(dataset_module)
    dataset_class = getattr(module, dataset_class)
    return dataset_class(**train_config.data.custom_dataset.args)  # type: ignore


def extract_module_and_class(name: str) -> Tuple[Optional[str], str]:
    """Extracts the module and class name from a full name. Returns None for the module if the name does not contain a dot.

    :param name: Full name of a class or function, e.g. "foo.bar.Baz"
    :return: Tuple of module and class name, e.g. ("foo.bar", "Baz") if name is "foo.bar.Baz"
    """
    class_module = name.split(".")
    if len(class_module) < 2:
        return None, class_module[0]
    return ".".join(class_module[:-1]), class_module[-1]
