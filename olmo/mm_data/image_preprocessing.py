from typing import Tuple

import numpy as np
from PIL import Image

from olmo.mm_data.image_token_size import ImageTokenSizer, FixedNumberOfToken


class ImagePreprocessor:

    def image_token_sizer(self) -> ImageTokenSizer:
        """Return an algorithm for computing number of tokens per an image"""
        raise NotImplementedError()

    def __call__(self, image: Image.Image, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        """Converts an image into patches

        image: PIL image to convert
        offset: offset of the image into the input tokens

         Returns:
            image: [n_patch, h, w, embed_dim] tensor
            offsets: [n_patches] token offsets of the individual patches

        n_patch can change between different images, but other dimensions must be fixed
        """
        raise NotImplementedError()


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

    def __call__(self, image: Image.Image, offset):
        image_w, image_h = self.image_size
        patch_w, patch_h = self.patch_size
        n_w_patches = image_w // patch_w
        n_h_patches = image_h // patch_h
        image = image.convert("RGB")
        if self.resize_method == "bicubic":
            method = Image.BICUBIC
        else:
            raise NotImplementedError(self.resize_method)
        image = image.resize(self.image_size, method)
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.reshape(image, [n_w_patches, patch_w, n_h_patches, patch_h, 3])
        image = np.transpose(image, [0, 2, 1, 3, 4])
        image = np.reshape(image, [n_w_patches*n_h_patches, patch_w, patch_h, 3])
        return image, np.arange(offset, offset+self.n_tokens, dtype=np.int32)
