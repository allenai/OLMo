import math
from typing import Union, Tuple

import numpy as np


class ImageTokenSizer:
    """Computes the number of tokens an image will be transformed into"""

    def __call__(self, width: Union[int, np.ndarray], height: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
        """Number of tokens for an image of width x height

        If the inputs are numpy arrays, this should compute the sizes all the images and return a scalar
        or an array of the same size as the inputs
        """
        raise NotImplementedError()

    def get_id(self) -> str:
        """Persistent ID that reflects the behaviour of `self.num_tokens`"""
        raise NotImplementedError()


class FixedNumberOfToken(ImageTokenSizer):
    def __init__(self, tokens: int):
        self.tokens = tokens

    def __call__(self, width: int, height: int) -> int:
        return self.tokens

    def get_id(self) -> str:
        return f"fixed{self.tokens}"


class PerPatch(ImageTokenSizer):
    def __init__(self, patch_h, patch_w):
        self.patch_h = patch_h
        self.patch_w = patch_w

    def __call__(self, width: int, height: int) -> int:
        n_w = (width + self.patch_w - 1) // self.patch_w
        n_h = (height + self.patch_h - 1) // self.patch_h
        return n_w*n_h

    def get_id(self) -> str:
        return f"per-patch{self.patch_h}x{self.patch_w}"


def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    excerpt from https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def select_best_resolution_vec(width, height, possible_resolutions):
    """Vectorized `select_best_resolution`"""
    original_size = np.stack([width, height], -1)
    original_res = np.prod(original_size, -1)
    possible_resolutions = np.array(possible_resolutions)  # [n_res, 2]
    scale = np.min(possible_resolutions[None, :] / original_size[:, None], -1)  # [n_images, n_res]
    downscaled = (scale[:, :, None]*original_size[:, None]).astype(np.int64)
    downscaled_res = np.prod(downscaled, -1)  # [n_images, n_res]
    # [n_examples, n_res]
    effective_resolution = np.minimum(downscaled_res, original_res[:, None])  # [n_images, n_res]
    wasted_resolution = original_res[:, None] - effective_resolution
    # Sort by effective_resolution, but with wasted_resolution as the tie breaker
    res_order = effective_resolution * (wasted_resolution.max()+1) + wasted_resolution
    selected = np.argmax(res_order, -1)  # [n_images]
    return possible_resolutions[selected].T


class AnyResImageTokenizer(ImageTokenSizer):
    def __init__(self, target_w, target_h, patch_w, patch_h, possible_resolutions, resample_tokens=None):
        assert target_w % patch_w == 0
        assert target_h % patch_h == 0
        self.target_w = target_w
        self.target_h = target_h
        self.patch_w = patch_w
        self.patch_h = patch_h
        assert all([res[0] % target_w == 0 and res[1] % target_h == 0 for res in possible_resolutions])
        self.possible_resolutions = possible_resolutions
        self.use_resampler = resample_tokens is not None
        self.n_tokens = resample_tokens or (target_w // patch_w) * (target_h // patch_h)

    def get_grid_shape(self, width: int, height: int) -> Tuple[int, int]:
        width, height = select_best_resolution((width, height), self.possible_resolutions)
        return width // self.target_w, height // self.target_h

    def get_unpad_shape(self, image_size: Tuple[int, int], original_size: Tuple[int, int]) -> Tuple[int, int]:
        original_width, original_height = original_size
        current_width, current_height = image_size
        original_aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height
        if original_aspect_ratio > current_aspect_ratio:
            scale_factor = current_width / original_width
            new_height = int(original_height * scale_factor)
            padding = (current_height - new_height) // 2
            width = current_width
            height = current_height - 2 * padding
        else:
            scale_factor = current_height / original_height
            new_width = int(original_width * scale_factor)
            padding = (current_width - new_width) // 2
            width = current_width - 2 * padding
            height = current_height
        return width, height

    def __call__(self, width: int, height: int) -> int:
        if isinstance(width, np.ndarray):
            best_solution = select_best_resolution_vec(width, height, self.possible_resolutions)
        else:
            best_solution = select_best_resolution((width, height), self.possible_resolutions)
        if self.use_resampler:
            n_newlines = (best_solution[1] // self.target_h) * int(math.sqrt(self.n_tokens))
            return (1 + (best_solution[0] // self.target_w) * (best_solution[1] // self.target_h)) * self.n_tokens + n_newlines
        else:
            # Considering padding removal
            new_width, new_height = self.get_unpad_shape(
                (best_solution[0] // self.patch_w, best_solution[1] // self.patch_h),
                (width, height)
            )
            n_newlines = new_height
            return self.n_tokens + new_width * new_height + n_newlines

    def get_id(self) -> str:
        max_num_grids = 1 + max((w // self.target_w) * (h // self.target_h) for w, h in self.possible_resolutions)
        return f"anyres-{max_num_grids}-{self.n_tokens}"