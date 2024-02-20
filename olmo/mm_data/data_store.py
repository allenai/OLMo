import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Union, List, Tuple, Optional, Iterable, Iterator

import numpy as np
from PIL import Image

from olmo.mm_data.data_config import MMStorageConfig
from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.mm_data.object_store import ObjectStore
from olmo.util import get_bytes_range


@dataclass
class ImageChunk:
    """Image that has been uploaded to the object store"""
    object_id: bytes
    w: int
    h: int
    num_tokens: Optional[int]=None

    def byte_len(self):
        # 3 tokens of two bytes and object id
        return len(self.object_id) + 6

    def is_masked(self):
        return True

    def dump(self, data_fh, config: MMStorageConfig):
        data_fh.write(config.image_start_bytes)
        data_fh.write(np.array([self.w, self.h], dtype=np.uint16).tobytes())
        data_fh.write(self.object_id)


@dataclass
class TextChunk:
    """Tokenized and possibly masked text"""
    tokens: np.ndarray
    masked: bool

    @property
    def num_tokens(self):
        return len(self.tokens)

    def byte_len(self):
        return len(self.tokens)*2 + self.masked*4

    def is_masked(self):
        return self.masked

    def dump(self, data_fh, config: MMStorageConfig):
        if self.masked:
            data_fh.write(config.mask_start_bytes)
            data_fh.write(self.tokens.tobytes())
            data_fh.write(config.mask_end_bytes)
        else:
            data_fh.write(self.tokens.tobytes())


RawExample = List[Union[TextChunk, ImageChunk]]


class ExampleReader:
    """Reads examples stored in data files given a file_id, byte offset, and length"""

    def __init__(
            self,
            image_store: ObjectStore,
            image_sizer: ImageTokenSizer,
            data_files: Dict[int, str],
            storage_config: MMStorageConfig
    ):
        self.image_store = image_store
        self.image_sizer = image_sizer
        self.data_files = data_files
        self.storage_config = storage_config

    def get_raw(self, file_id, start_byte, num_bytes) -> List[RawExample]:
        buffer = get_bytes_range(self.data_files[file_id], start_byte, num_bytes)
        cfg = self.storage_config

        # TODO maybe byte regex would be faster? like:
        # re.compile(b"|".join([
        #     cfg.mask_start_bytes, cfg.mask_end_bytes, cfg.doc_end_bytes, cfg.image_start_bytes,
        # ])).finditer(buffer)

        data = np.frombuffer(buffer, np.uint16)
        is_special_token = (
            (data == cfg.image_start_token_id) |
            (data == cfg.mask_start_token_id) |
            (data == cfg.mask_end_token_id) |
            (data == cfg.document_end_token)
        )
        out = []
        parts = []
        on = 0
        is_masked = False
        for ix in np.argwhere(is_special_token)[:, 0]:
            if ix < on:
                # special token occurred inside an object id, skip it
                continue
            marker_token = data[ix]
            if on != ix:
                # Add text that occurred before this special token
                parts.append(TextChunk(data[on:ix], marker_token == cfg.mask_end_token_id))
            on = ix + 1
            if marker_token == cfg.image_start_token_id:
                w, h = buffer[on:on+2]
                on += 2
                image_id = buffer[on*2:on*2 + cfg.object_id_length]
                parts.append(ImageChunk(image_id, w, h, self.image_sizer(w, h)))
                on += cfg.object_id_length // 2
            elif marker_token == cfg.document_end_token:
                out.append(parts)
                parts = []

        if on != len(data):
            parts.append(TextChunk(data[on:], False))
        if parts:
            out.append(parts)
        return out

    def _build_sequence(
        self,
        chunks: List[RawExample],
        sequence_length=None,
        return_segments=False
    ) -> Dict[str, np.ndarray]:
        if sequence_length is None:
            sequence_length = sum(sum(y.num_tokens for y in x) for x in chunks)
        indices = np.zeros(sequence_length, np.uint16)
        mask = np.ones(sequence_length, np.bool_)
        images = []
        offsets = []
        if return_segments:
            segments = np.zeros(sequence_length, np.int32)
        else:
            segments = None

        total_tokens = 0
        for segment_id, parts in enumerate(chunks, start=1):
            start_token = total_tokens
            for part in parts:
                if isinstance(part, TextChunk):
                    # Text might get truncated
                    token_end = min(total_tokens+part.num_tokens, sequence_length) - total_tokens
                    indices[total_tokens:total_tokens+part.num_tokens] = part.tokens[:token_end]
                    if not part.is_masked():
                        mask[total_tokens:total_tokens+part.num_tokens] = False
                else:
                    offsets.append(total_tokens)
                    image = Image.open(BytesIO(self.image_store.get(part.object_id)))
                    images.append(np.array(image.convert('RGB')))
                total_tokens += part.num_tokens
            if return_segments:
                segments[start_token:total_tokens] = segment_id

        out = dict(
            input_ids=indices,
            label_mask=mask,
            image_offsets=np.asarray(offsets, np.int32) if offsets else np.zeros((0,), np.int32),
            images=images
        )
        if return_segments:
            out["segment_ids"] = segments
        return out

    def read_ranges(
        self,
        examples: List[Tuple[int, int, int]],
        sequence_length: Optional[int]=None,
        return_segments=True
    ) -> Dict:
        """Load a list of examples as a single sequence of length `sequence_length`

        examples: list of (file_id, start_byte, num_bytes) tuples
        sequence_length: max output sequence length, examples will be truncated if needed.
                         If truncation leaves masked tokens/images at the ends of they will be
                         removed since they provide no training signal
                         It is an error if a document is entirely removed
        return_segments: return segments ids in the output
        """
        total_tokens = 0
        all_chunks = []
        for segment_id, (file_id, start_byte, num_bytes) in enumerate(examples):
            start_token = total_tokens
            examples = self.get_raw(file_id, start_byte, num_bytes)
            for example in examples:
                if sequence_length:
                    new_tokens = sum(x.num_tokens for x in example)
                    # Possibly truncate this example, we remove chunks until we have a non-masked
                    # chunk that has at least some tokens that do not need to get truncated. This
                    # means we could truncate the document and then add in a new document after it
                    while example and (
                        example[-1].is_masked() or
                        (new_tokens + total_tokens - example[-1].num_tokens) > sequence_length
                    ):
                        new_tokens -= example[-1].num_tokens
                        example.pop()
                    if not example:
                        # The shuffling mechanism should ensure this never happens
                        raise ValueError("An example got completely truncated")
                    total_tokens += new_tokens
                all_chunks.append(example)

        return self._build_sequence(all_chunks, sequence_length, return_segments)

    def read_range(self, file_id, start_byte, num_bytes,
                   sequence_length: int=None, return_segments=True):
        return self.read_ranges([(file_id, start_byte, num_bytes)], sequence_length, return_segments)


def build_data_file(
    examples: Iterable[RawExample],
    data_file: str,
    data_config: MMStorageConfig
) -> Iterator[Tuple[int, int, RawExample]]:
    """Builds a datafile and yield the saved examples and their locations"""
    with open(data_file, "wb") as data_fh:
        on_byte = 0
        for example in examples:
            example_start = on_byte
            for chunk in example:
                chunk.dump(data_fh, data_config)
                on_byte += chunk.byte_len()

            example_length = on_byte-example_start
            data_fh.write(data_config.doc_end_bytes)
            on_byte += 2

            # yield back example and its location in case the indexer needs it
            yield example_start, example_length, example

