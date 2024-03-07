import io
from dataclasses import dataclass
from typing import Union, List

import PIL.Image
import numpy as np
from PIL import Image

from olmo import Tokenizer
from olmo.mm_data.data_store import ImageChunk, Document, TextChunk
from olmo.util import read_file

try:
    import imagesize
except ImportError:
    imagesize = None


@dataclass
class Masked:
    text: str


@dataclass
class ImageFile:
    image_file: str


@dataclass
class ImageData:
    bytes: bytes
    w: int
    h: int


InputImageTypes = Union[ImageFile, Image.Image, np.ndarray, bytes]
InputExample = List[Union[str, Masked, InputImageTypes]]


def get_image_size(data: bytes):
    if imagesize is None:
        pil_image = Image.open(io.BytesIO(data))
        return pil_image.size
    else:
        return imagesize.get(io.BytesIO(data))


def _preprocess_image(image, object_store, data_config) -> ImageChunk:
    """Ensure the image is stored and get the image id, height, and width"""
    if isinstance(image, (Image.Image, np.ndarray)):
        # Raw image, assume a jpeg encoding
        if isinstance(image, np.ndarray):
            image = PIL.Image.fromarray(image)
        w, h = image.width, image.height
        data = io.BytesIO()
        image.save(data, "jpeg")
        data = data.getbuffer()
        object_id = data_config.get_object_id(data)
        if not object_store.contains(object_id):
            object_store.write(object_id, data)
        return ImageChunk(object_id, w, h)
    elif isinstance(image, bytes):
        object_id = data_config.get_object_id(image)
        if not object_store.contains(object_id):
            object_store.write(object_id, image)
        return ImageChunk(object_id, *get_image_size(image))
    elif isinstance(image, ImageFile):
        data = read_file(image.image_file)
        object_id = data_config.get_object_id(data)
        if not object_store.contains(object_id):
            object_store.write(object_id, data)
        out = ImageChunk(object_id, *get_image_size(data))
        return out
    else:
        raise NotImplementedError(type(image))


def preprocess_example(
        example: InputExample, tokenizer: Tokenizer, object_store, data_config, add_bos_token: bool = False, add_eos_token: bool = True,
) -> List[Document]:
    """pre-processing examples by tokenizing the text and storing the images"""
    out = []
    text = []
    is_masked = None
    prev_was_text = False
    last_text_ix = max(i for i, ex in enumerate(example) if isinstance(ex, (Masked, str)))
    first_text_ix = min(i for i, ex in enumerate(example) if isinstance(ex, (Masked, str)))
    assert not (add_bos_token and add_eos_token)

    if not example:
        raise ValueError("Zero length example")

    for ix, chunk in enumerate(example):
        if isinstance(chunk, (str, Masked)):
            if isinstance(chunk, str):
                if prev_was_text and not is_masked:
                    raise ValueError("Consecutive text spans")
                is_masked = False
                text = chunk
            else:
                if prev_was_text and is_masked:
                    raise ValueError("Consecutive masked text spans")
                is_masked = True
                text = chunk.text
            if not text:
                raise ValueError("Got empty text")
            if prev_was_text and not text[0].isspace():
                # (hopefully) ensures tokens could never cross masked/unmasked boundaries
                # We assume the tokenizer never merges whitespace with preceding text
                # TODO just assuming this is a bit dangerous, should we check this more
                # carefully by tokenizing the text as a single string?
                raise ValueError("Text must start with a space, or be preceded by an image, "
                                 "or start the document")
            ends_with_space = text[-1].isspace()
            add_special_tokens = (add_bos_token and ix == first_text_ix) or (add_eos_token and ix == last_text_ix)
            tokens = np.array(tokenizer.encode(text, add_special_tokens=add_special_tokens), np.uint16)
            if not is_masked and add_bos_token and ix == first_text_ix:
                assert tokens[0] == tokenizer.bos_token_id
                out.append(TextChunk(tokens[:1], True))
                out.append(TextChunk(tokens[1:], False))
            else:
                out.append(TextChunk(tokens, is_masked))
            prev_was_text = True
        else:
            prev_was_text = False
            out.append(_preprocess_image(chunk, object_store, data_config))
    if out[-1].is_masked():
        raise ValueError("Examples should not end with a masked span")
    return out