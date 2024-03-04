import io
from pathlib import Path

import numpy as np
from PIL import Image

from olmo.mm_data.data_store import ExampleReader, TextChunk, ImageChunk, write_data_file, MMStorageConfig
from olmo.mm_data.image_token_size import FixedNumberOfToken
from olmo.mm_data.object_store import InMemoryStore


def test_example_store_text(tmp_path: Path):
    data = np.array([51, 6, 1, 3], dtype=np.uint16)
    data_file = tmp_path.joinpath("data.bin")
    with open(data_file, "wb") as f:
        f.write(data.tobytes())
    store = ExampleReader({0: data_file}, None, None, MMStorageConfig())
    out = store.read_range(0, 0, len(data) * 2)
    assert np.all(out["input_ids"] == data)
    assert np.all(out["label_mask"])

    out = store.read_range(0, 0, 2)
    assert np.all(out["input_ids"] == data[:1])

    out = store.read_range(0, 2, len(data) * 2)
    assert np.all(out["input_ids"] == data[1:])

    out = store.read_ranges([(0, 0, 6)], 12)
    indices = out["input_ids"]
    assert np.all(indices[:3] == data[:3])
    assert np.all(indices[3:] == 0)
    assert indices.shape == (12,)

    out = store.read_ranges([(0, 2, 8)], 1)["input_ids"]
    assert np.all(out == data[1:2])

    out = store.read_ranges([(0, 0, 8), (0, 2, 2), (0, 2, 6)], 7, return_segments=True)
    indices = out["input_ids"]
    assert np.all(indices[:4] == data)
    assert np.all(indices[4:5] == data[1:2])
    assert np.all(indices[5:] == data[1:3])
    assert np.all(out["segment_ids"] == np.array([1] * 4 + [2] + [3] * 2))


def test_example_store_two_files(tmp_path: Path):
    data1 = np.array([51, 6, 1, 3], dtype=np.uint16)
    data_file1 = tmp_path.joinpath("data1.bin")
    with open(data_file1, "wb") as f:
        f.write(data1.tobytes())
    data2 = np.array([3, 2, 123], dtype=np.uint16)
    data_file2 = tmp_path.joinpath("data2.bin")
    with open(data_file2, "wb") as f:
        f.write(data2.tobytes())
    store = ExampleReader({1: data_file1, 2: data_file2}, None, None, MMStorageConfig())

    assert np.all(store.read_range(1, 2, 6)["input_ids"] == data1[1:4])
    assert np.all(store.read_range(2, 0, 4)["input_ids"] == data2[:2])

    out = store.read_ranges([
        (1, 0, 2), (1, 2, 4), (2, 2, 4),
        (1, 0, 2), (2, 4, 6)
    ], 9)
    assert np.all(out["input_ids"] == np.array([51, 6, 1, 2, 123, 51, 123, 0, 0], dtype=np.uint16))


def _tokens(*data):
    return TextChunk(np.array(data, np.uint16), False)


def _mtokens(*data):
    return TextChunk(np.array(data, np.uint16), True)


def test_example_store_mm(tmp_path: Path):
    cfg = MMStorageConfig()
    ms = cfg.mask_start_token_id
    me = cfg.mask_end_token_id
    eod = cfg.document_end_token
    im = cfg.image_start_token_id
    object_id = [1, 3, ms, me, eod, im]  # test when this includes special tokens
    object_id += [0] * (cfg.object_id_length // 2 - len(object_id))
    image_tok = ImageChunk(np.array(object_id, np.uint16).tobytes(), 4, 4)

    object_store = InMemoryStore()
    image_ar = np.random.RandomState(1).randint(0, 255, (2, 2, 3), np.uint8)
    image = Image.fromarray(image_ar)
    image_data = io.BytesIO()
    image.save(image_data, "png")
    object_store.write(np.array(object_id, np.uint16).tobytes(), image_data.getbuffer())

    data = [
        [_tokens(8, 3, 1)],
        [_tokens(71, 12), _mtokens(39)],
        [_tokens(3, 3), image_tok, image_tok, _tokens(3)],
        [image_tok, _tokens(9), _mtokens(8), image_tok, _tokens(11)]
    ]
    data_file = tmp_path.joinpath("data1.bin")
    indices = [(0, x[0], sum(c.byte_len() for c in x[1])) for x in write_data_file(data, data_file, cfg)]

    store = ExampleReader({0: data_file}, object_store, FixedNumberOfToken(4), cfg)

    ex1 = store.read_range(*indices[0])
    assert np.all(ex1["input_ids"] == np.array([8, 3, 1]))
    assert np.all(ex1["label_mask"])
    assert ex1["images"] == []

    ex2 = store.read_range(*indices[1])
    assert np.all(ex2["input_ids"] == np.array([71, 12, 39]))
    assert np.all(ex2["label_mask"] == np.array([True, True, False]))

    ex3 = store.read_range(*indices[2])
    assert np.all(ex3["input_ids"] == np.array([3, 3] + [0] * 8 + [3]))
    assert np.all(np.array(ex3["images"][0]) == image_ar)
    assert np.all(np.array(ex3["images"][1]) == image_ar)
    assert len(ex3["images"]) == 2
    assert np.all(ex3["image_offsets"] == np.array([2, 6]))

    ex4 = store.read_range(*indices[3])
    assert np.all(ex4["input_ids"] == np.array([0] * 4 + [9, 8] + [0] * 4 + [11]))
    assert np.all(ex4["label_mask"] == ~((ex4["input_ids"] == 8) | (ex4["input_ids"] == 0)))
    assert np.all(ex4["image_offsets"] == np.array([0, 6]))

    first3 = store.read_range(0, 0, indices[3][1], return_segments=True)
    assert np.all(first3["input_ids"] ==
                  np.concatenate([ex1["input_ids"], ex2["input_ids"], ex3["input_ids"]]))
    assert np.all(first3["label_mask"] ==
                  np.concatenate([ex1["label_mask"], ex2["label_mask"], ex3["label_mask"]]))
    assert len(first3["images"]) == 2
    for ix in first3["image_offsets"]:
        assert np.all(first3["input_ids"][ix:ix + 4] == 0)
    assert np.all(first3["segment_ids"] == np.array([1] * 3 + [2] * 3 + [3] * 11))
