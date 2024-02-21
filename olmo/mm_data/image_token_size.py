import torch
from PIL.Image import Image
import numpy as np


class ImageTokenSizer:
    """Computes the number of tokens an image will be transformed into"""

    def __call__(self, width: int, height: int) -> int:
        """Number of tokens for an image of width x height"""
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


class ResizeImage(ImagePreprocessor):
    def __init__(self, image_size, patch_size, resize_method):
        self.image_size = image_size
        self.resize_method = resize_method
        self.patch_size = patch_size
        assert image_size[0] % patch_size[0] == 0
        assert image_size[1] % patch_size[1] == 0
        w_tok, h_tok = np.array(image_size) // np.array(patch_size)
        self.n_tokens = w_tok*h_tok

    def image_token_sizer(self) -> ImageTokenSizer:
        return FixedNumberOfToken(self.n_tokens)

    def __call__(self, image):
        image_w, image_h = self.image_size
        patch_w, patch_h = self.patch_size
        n_w_patches = image_w // patch_w
        n_h_patches = image_h // patch_h
        image = image.resize(self.image_size, self.resize_method)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.permute(image, [image_w//patch_h, patch_h, image_w//patch_h,])
        image = np.reshape(image, [1, n_w_patches*n_w_patches, 3*patch_w*patch_h])
        return torch.as_tensor(image)

