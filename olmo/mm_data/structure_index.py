from typing import List, Optional, Union, Iterator

import numpy as np

from olmo.mm_data.data_store import ExampleReader, TextChunk, Document, read_data_file, MMStorageConfig
from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.util import get_bytes_range, NumpyList, read_file


def get_index_file(data_file):
    """Expected location of an index file for a data file"""
    return data_file + ".index"


"""DType for meta-data we might need to shuffle/split/pack document"""
DOCUMENT_INFO = np.dtype([
    ("start_byte", np.uint64),  # Starting location of the document
    ("num_bytes", np.uint32),  # Byte length of the document
    ("num_tokens", np.uint32),  # Number of tokens of this document
    ("pure_text", np.bool_),  # Is this data unmasked text (and therefore can be split)
])
# To split multi-modal document we would need token/byte/pure_text data for each
# sub-section of the example so we (1) know where we can split and (2) know the correct lengths
# of the halves. This is possible but complicates indexing quite a bit so for now
# we don't support it


class Indexer:
    """Writes an index file that can be read to quickly get meta-data about each document in a data file"""
    def write_index(self, index_file: str, iterator):
        """Consume examples from `iterator` and build the index"""
        raise NotImplementedError()

    def get_indices(
        self,
        data_file: str,
        image_size: ImageTokenSizer,
        storage_config: MMStorageConfig,
        index_file: str,
    ) -> np.ndarray:
        """Read the index and return an array of dtype `DOCUMENT_INFO`"""
        raise NotImplementedError()


def get_example_info(start_byte, ex: Document):
    num_byte, num_token = 0, 0
    pure_text = True
    for c in ex:
        num_byte += c.byte_len()
        num_token += c.num_tokens
        pure_text &= isinstance(c, TextChunk) & (not c.is_masked())
    return start_byte, num_byte, num_token, pure_text


class NullIndexer(Indexer):
    """No indexing, requires re-reading the entire data file to get indices"""

    def get_indices(self, data_file, sizer, storage_config, index_file: str=None) -> Iterator:
        examples = read_data_file(data_file, 0, -1, storage_config, sizer)
        on = 0
        data = np.rec.array(len(examples), dtype=DOCUMENT_INFO)
        for i, ex in enumerate(examples):
            data[i] = get_example_info(on, ex)
            on += data[i].num_bytes + 2
        return data

    def write_index(self, index_file, iterator):
        for _ in iterator:
            pass


def lengths_to_segments(lengths: np.ndarray) -> np.ndarray:
    """Equivalent of [0]*lengths[0] + [1]*lengths[1].... [n]*lengths[n]"""
    # For millions of lengths, this tricky cumulative sum method is significantly faster
    # than using a python loop (over 4x)
    ends = np.trim_zeros(lengths, trim="b")
    ends = np.cumsum(ends)
    idx = np.zeros(ends[-1], dtype=np.int64)
    np.add.at(idx, ends[:-1], 1)
    np.cumsum(idx, out=idx)
    return idx
    # slower:
    # n = lengths.sum()
    # out = np.zeros(n, dtype=lengths.dtype)
    # on = 0
    # for i, l in enumerate(lengths):
    #     if l:
    #         out[on:on+l] = i
    #         on += l
    # return out


class VectorizedIndexer(Indexer):
    """Index optimized for vectorized parsing"""
    VERSION = 1

    DTYPE = np.dtype([
        ("num_text_tokens", np.uint32),
        ("num_masks", np.uint8),
        ("num_images", np.uint8),
    ])

    def get_indices(self, data_file, image_sizer, cfg: MMStorageConfig, index_file: str=None):
        if index_file is None:
            index_file = get_index_file(data_file)
        buf = read_file(index_file)
        image_byte_size = 6 + cfg.object_id_length

        file_version = np.frombuffer(buf[:4], np.uint32)
        if file_version != self.VERSION:
            raise ValueError()

        # Read the data from the file
        num_images = int(np.frombuffer(buf[-8:], np.uint64))
        data = np.frombuffer(buf[4:-8-num_images*4], dtype=self.DTYPE)
        image_sizes = np.frombuffer(buf[-8-num_images*4:-8], dtype=np.uint16).reshape((-1, 2))
        num_masks = data["num_masks"].astype(np.uint32)
        num_images = data["num_images"].astype(np.uint32)
        num_text_tokens = data["num_text_tokens"]

        out = np.zeros(len(num_text_tokens), dtype=DOCUMENT_INFO)
        out["pure_text"] = np.logical_and(num_images == 0, num_masks == 0)

        # byte length of each example is derived from the other fields
        byte_lens = data["num_text_tokens"]*2
        byte_lens += data["num_masks"]*4
        byte_lens += data["num_images"]*image_byte_size
        out["num_bytes"] = byte_lens

        # start byte by doing a cumulative sum on the lengths starting at 0
        start_bytes = np.zeros(len(byte_lens), np.uint64)
        start_bytes[1:] = byte_lens[:-1]
        start_bytes[1:] += 2  # for the doc start tokens
        np.cumsum(start_bytes, out=out["start_byte"])

        # number of tokens by batch-computing the image token sizes and adding to the num text tokens
        num_tokens = num_text_tokens
        image_sizes = image_sizes.astype(np.int32)  # Just in case the image size doesn't low/unsigned types
        image_tokens = image_sizer(image_sizes[:, 0], image_sizes[:, 1])
        num_images = data["num_images"].astype(np.int64)
        image_segments = lengths_to_segments(num_images)  # build mapping of image -> document it belongs to
        np.add.at(num_tokens, image_segments, image_tokens)
        out["num_tokens"] = num_tokens
        return np.rec.array(out, copy=False)

    def write_index(self, index_file, iterator):
        image_sizes = NumpyList(np.uint16)
        with open(index_file, "wb") as fh:
            fh.write(np.array(self.VERSION, np.uint32).tobytes())
            for ex in iterator:
                n_text = 0
                num_masked = 0
                num_images = 0
                for chunk in ex:
                    if isinstance(chunk, TextChunk):
                        if chunk.is_masked():
                            num_masked += 1
                        n_text += chunk.num_tokens
                    else:
                        num_images += 1
                        image_sizes.append(chunk.w)
                        image_sizes.append(chunk.h)
                header = (n_text, num_masked, num_images)
                header_ar = np.array(header, self.DTYPE)[()]
                assert header == tuple(header_ar)  # check for overflows
                fh.write(header_ar.tobytes())
            # Save images and total image count at the end of the file so we can load and
            # parse it as a single [w, h] array
            image_sizes = image_sizes.to_array()
            fh.write(image_sizes.tobytes())
            fh.write(np.array(len(image_sizes)//2, np.uint64).tobytes())
