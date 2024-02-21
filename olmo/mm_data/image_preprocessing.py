from typing import Tuple

import numpy as np
from PIL import Image

from olmo.mm_data.image_token_size import ImageTokenSizer, FixedNumberOfToken


class ImagePreprocessor:
    """Does image-preprocessing for the vision backbone"""

    def image_token_sizer(self) -> ImageTokenSizer:
        """Return a function for computing number of tokens per an image"""
        raise NotImplementedError()

    def __call__(self, image: Image.Image, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        """Converts an image into patches

        image: PIL image to convert
        offset: offset of the image into the input tokens

         Returns:
            image: [n_patches, h, w, embed_dim] tensor
            offsets: [n_patches, n_tokens] token offsets for the output embeddings

        n_patch can change between different images, but other dimensions must be fixed
        """
        raise NotImplementedError()


class ResizeImage(ImagePreprocessor):
    """Resize an image and yield the individual patches"""
    def __init__(self, image_size, patch_size, resize_method, pad=True):
        self.image_size = image_size
        self.resize_method = resize_method
        self.patch_size = patch_size
        assert image_size[0] % patch_size[0] == 0
        assert image_size[1] % patch_size[1] == 0
        w_tok, h_tok = np.array(image_size) // np.array(patch_size)
        self.n_tokens = w_tok*h_tok
        self.pad = pad

    def image_token_sizer(self) -> ImageTokenSizer:
        return FixedNumberOfToken(self.n_tokens)

    def __call__(self, image: Image.Image, offset):
        patch_w, patch_h = self.patch_size
        target_w, target_h = self.image_size
        image = image.convert("RGB")
        if self.resize_method == "bicubic":
            method = Image.BICUBIC
        else:
            raise NotImplementedError(self.resize_method)

        if self.pad and image.width != image.height:
            w_r, h_r = np.array(image.size) / np.array(self.image_size)
            if w_r > h_r:
                h = round((target_w / image.width) * image.height)
                image = image.resize([target_w, h], method)
            else:
                w = round((target_h / image.height) * image.width)
                image = image.resize([w, target_h], method)
        else:
            image = image.resize(self.image_size, method)
        image = np.array(image, dtype=np.float32) / 255.0

        left_pad = target_w - image.shape[0]
        bot_pad = target_h - image.shape[1]
        if left_pad or bot_pad:
            image = np.pad(image, [[0, left_pad], [0, bot_pad], [0, 0]])
        assert image.shape == (target_w, target_h, 3)

        n_w_patches = target_w // patch_w
        n_h_patches = target_h // patch_h
        image = np.reshape(image, [n_w_patches, patch_w, n_h_patches, patch_h, 3])
        image = np.transpose(image, [0, 2, 1, 3, 4])
        image = np.reshape(image, [n_w_patches*n_h_patches, patch_w, patch_h, 3])
        return image, np.arange(offset, offset+self.n_tokens, dtype=np.int32)
