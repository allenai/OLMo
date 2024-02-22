from typing import Tuple

import numpy as np
from PIL import Image
import functools
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from olmo.mm_data.image_token_size import ImageTokenSizer, FixedNumberOfToken
from olmo.mm_data.image_util import convert_image_to_rgb, to_numpy_array, normalize

OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


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
        if self.resize_method == "bicubic":
            method = Image.BICUBIC
        else:
            raise NotImplementedError(self.resize_method)

        image = image.convert("RGB")
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
        image = np.array(image, dtype=np.float32) / 255.0 # height/width transpose from pil_image to ndarray: (w, h) -> (h, w, c)

        left_pad = target_w - image.shape[1]
        bot_pad = target_h - image.shape[0]
        if left_pad or bot_pad:
            image = np.pad(image, [[0, bot_pad], [0, left_pad], [0, 0]])
        assert image.shape == (target_h, target_w, 3)

        n_h_patches = target_h // patch_h
        n_w_patches = target_w // patch_w
        image = np.reshape(image, [n_h_patches, patch_h, n_w_patches, patch_w, 3])
        image = np.transpose(image, [0, 2, 1, 3, 4])
        image = np.reshape(image, [n_h_patches*n_w_patches, patch_h, patch_w, 3])
        return image, np.arange(offset, offset+self.n_tokens, dtype=np.int32)
    

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        convert_image_to_rgb,
        functools.partial(to_numpy_array, to_float=True),
        functools.partial(normalize, mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD),
    ])


def expand2square(pil_img, background_color=OPENAI_CLIP_MEAN):
    # Excerpt from https://github.com/haotian-liu/LLaVA/blob/main/llava/mm_utils.py
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class ClipImageResize(ImagePreprocessor):
    """LLaVA + CLip Image Resize"""
    def __init__(self, image_size, patch_size, resize_method, pad=False):
        self.image_size = image_size # (width, height)
        assert resize_method == "bicubic"
        self.patch_size = patch_size # (w, h)
        assert image_size[0] == image_size[1]
        assert patch_size[0] == patch_size[1]
        assert image_size[0] % patch_size[0] == 0
        w_tok, h_tok = np.array(image_size) // np.array(patch_size)
        self.n_tokens = w_tok*h_tok
        self.pad = pad

    def image_token_sizer(self) -> ImageTokenSizer:
        return FixedNumberOfToken(self.n_tokens)

    def __call__(self, image: Image.Image, offset):
        patch_w, patch_h = self.patch_size
        target_w, target_h = self.image_size
        n_w_patches = target_w // patch_w
        n_h_patches = target_h // patch_h
        if self.pad:
            image = expand2square(image, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
        image = _transform(self.image_size[0])(image)

        assert image.shape == (target_h, target_w, 3)
        image = np.reshape(image, [n_h_patches, patch_h, n_w_patches, patch_w, 3])
        image = np.transpose(image, [0, 2, 1, 3, 4])
        image = np.reshape(image, [n_h_patches*n_w_patches, patch_h, patch_w, 3])
        return image, np.arange(offset, offset+self.n_tokens, dtype=np.int32)