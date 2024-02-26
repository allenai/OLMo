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

from olmo.mm_data.image_token_size import ImageTokenSizer, FixedNumberOfToken, AnyResImageTokenizer
from olmo.mm_data.image_token_size import select_best_resolution
from olmo.mm_data.image_util import convert_image_to_rgb, to_numpy_array, normalize
from olmo.mm_data.image_util import resize_and_pad_image, divide_to_patches

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
    def __init__(self, image_size, patch_size, pad_image=False):
        self.image_size = image_size # (width, height)
        self.patch_size = patch_size # (w, h)
        assert image_size[0] == image_size[1]
        assert patch_size[0] == patch_size[1]
        assert image_size[0] % patch_size[0] == 0
        w_tok, h_tok = np.array(image_size) // np.array(patch_size)
        self.n_tokens = w_tok*h_tok
        self.pad_image = pad_image
        self.transform = _transform(self.image_size[0])

    def image_token_sizer(self) -> ImageTokenSizer:
        return FixedNumberOfToken(self.n_tokens)

    def __call__(self, image: Image.Image, offset):
        target_w, target_h = self.image_size
        if self.pad_image:
            image = expand2square(image, tuple(int(x*255) for x in OPENAI_CLIP_MEAN))
        image = self.transform(image)

        assert image.shape == (target_h, target_w, 3)
        image = np.transpose(image, [2, 0, 1])[None] # CHANNEL FIRST
        return image, np.arange(offset, offset+self.n_tokens, dtype=np.int32)


class AnyResClipImageResize(ImagePreprocessor):
    """LLaVA + CLip Image Resize"""
    def __init__(self, image_size, patch_size, possible_resolutions, resample_tokens=None):
        self.image_size = image_size # (width, height)
        self.patch_size = patch_size # (w, h)
        assert image_size[0] % patch_size[0] == 0
        assert image_size[1] % patch_size[1] == 0
        assert all([res[0] % self.image_size[0] == 0 and res[1] % self.image_size[1] == 0 for res in possible_resolutions])
        self.possible_resolutions = possible_resolutions
        self.resample_tokens = resample_tokens
        self.transform = _transform(image_size[0])
        self.image_sizer = AnyResImageTokenizer(
            self.image_size[0], self.image_size[1],
            self.patch_size[0], self.patch_size[1],
            self.possible_resolutions, self.resample_tokens)

    def image_token_sizer(self) -> ImageTokenSizer:
        return AnyResImageTokenizer(
            self.image_size[0], self.image_size[1],
            self.patch_size[0], self.patch_size[1],
            self.possible_resolutions, self.resample_tokens)

    def __call__(self, image: Image.Image, offset):
        target_w, target_h = self.image_size
        best_resolution = select_best_resolution(image.size, self.possible_resolutions)
        image_padded = resize_and_pad_image(image, best_resolution)
        patches = divide_to_patches(image_padded, target_w)
        downsampled_image = image.resize((target_w, target_h), Image.BICUBIC)

        image_patches = [downsampled_image] + patches
        image_patches = [np.transpose(self.transform(x), [2, 0, 1]) for x in image_patches] # CHANNEL FIRST
        assert all(x.shape == (3, target_h, target_w) for x in image_patches)
        image_patches = np.stack(image_patches, axis=0)
        n_tokens = self.image_sizer(image.size[0], image.size[1])
        return image_patches, np.arange(offset, offset+n_tokens, dtype=np.int32)