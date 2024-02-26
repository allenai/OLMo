from dataclasses import dataclass
from typing import Iterator

import numpy as np

from olmo.mm_data.data_store import ExampleReader, TextChunk, Document
from olmo.util import get_bytes_range


def get_index_file(data_file):
    return data_file + ".index"


@dataclass
class ExampleInfo:
    """meta-data about an example needed when shuffling/chunking"""

    start_byte: int
    """Starting location of the example"""

    num_bytes: int
    """Byte length of the example"""

    num_tokens: int
    """Number of tokens of this example"""

    pure_text: bool
    """Is this data unmasked text (and therefore can be split)"""

    # to split multi-modal examples we would need token/byte/pure_text data for each
    # sub-section of the example so we (1) know where we can split and (2) know the correct lengths
    # of the halves. This is possible but complicates indexing quite a bit so for now
    # we don't support it

    @staticmethod
    def from_example(start_byte, ex: Document):
        num_byte, num_token = 0, 0
        pure_text = True
        for c in ex:
            num_byte += c.byte_len()
            num_token += c.num_tokens
            pure_text &= isinstance(c, TextChunk) & (not c.is_masked())
        return ExampleInfo(start_byte, num_byte, num_token, pure_text)


class Indexer:
    """Writes an index file that can be read to quickly get structural information about each example"""

    def write_index(self, index_file: str, iterator):
        """Consume examples from `iterator` and build the index"""
        raise NotImplementedError()

    def get_indices(
            self,
            index_file: str,
            file_id: int,
            reader: ExampleReader,
    ) -> Iterator[ExampleInfo]:
        """Read the index"""
        raise NotImplementedError()


def get_indices_from_data_file(file_id, reader):
    examples = reader.get_raw(file_id, 0, -1)
    on = 0
    for ex in examples:
        info = ExampleInfo.from_example(on, ex)
        on += info.num_bytes + 2
        yield info


class NullIndexer(Indexer):
    """No indexing, requires re-reading the entire data file to get indices"""

    def get_indices(self, index_file: str, file_id, reader: ExampleReader) -> Iterator:
        return get_indices_from_data_file(file_id, reader)

    def write_index(self, index_file, iterator):
        for _ in iterator:
            pass


class BasicIndexer(Indexer):
    VERSION = 0

    DTYPE = np.dtype([
        ("num_text_tokens", np.uint32),
        ("num_masks", np.uint8),
        ("num_images", np.uint8),
    ])

    def get_indices(self, index_file, file_id, reader):
        data = get_bytes_range(index_file, 0, -1)
        file_version = np.frombuffer(data[:4], np.uint32)
        assert file_version == self.VERSION

        sizer = reader.image_sizer
        image_byte_size = 6 + reader.storage_config.object_id_length
        on = 4
        data_file_start_byte = 0
        while on < len(data):
            num_text_tokens, num_masks, num_images = np.frombuffer(
                data[on:on + self.DTYPE.itemsize], dtype=self.DTYPE)[0]
            on += self.DTYPE.itemsize
            num_tokens = num_text_tokens
            num_bytes = num_text_tokens * 2 + num_masks * 4
            if num_images:
                image_sizes = np.frombuffer(data[on:on + num_images * 4], dtype=np.uint16).reshape(-1, 2)
                on += num_images * 4
                num_bytes += image_byte_size * num_images
                for w, h in image_sizes:
                    num_tokens += sizer(w, h)
            pure_text = num_images == 0 and num_masks == 0
            yield ExampleInfo(data_file_start_byte, num_bytes, num_tokens, pure_text)
            data_file_start_byte += num_bytes + 2

    def write_index(self, index_file, iterator):
        with open(index_file, "wb") as fh:
            fh.write(np.array(self.VERSION, np.uint32).tobytes())
            for _, _, ex in iterator:
                n_text = 0
                num_masked = 0
                image_sizes = []
                for chunk in ex:
                    if isinstance(chunk, TextChunk):
                        if chunk.is_masked():
                            num_masked += 1
                        n_text += chunk.num_tokens
                    else:
                        image_sizes.append((chunk.w, chunk.h))
                header = (n_text, num_masked, len(image_sizes))
                header_ar = np.array(header, self.DTYPE)[()]
                assert header == tuple(header_ar)  # check for overflows
                fh.write(header_ar.tobytes())
                if image_sizes:
                    fh.write(np.array(image_sizes, np.uint16).tobytes())
