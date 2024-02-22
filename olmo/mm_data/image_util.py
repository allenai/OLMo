from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image


def convert_image_to_rgb(image):
    return image.convert("RGB")


def to_numpy_array(image: Image.Image, to_float: bool=True) -> np.ndarray:
    """Convert a PIL.Image.Image to a numpy array"""
    image = np.array(image)
    if to_float:
        image = image.astype(np.float32)
    return image

def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    """Normalize an image. image = (image - image_mean) / image_std."""
    num_channels = image.shape[-1]
    if not isinstance(mean, Iterable):
        mean = [mean] * num_channels
    if not isinstance(std, Iterable):
        std = [std] * num_channels
    mean, std = np.array(mean, dtype=image.dtype), np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image
