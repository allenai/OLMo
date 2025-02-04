import pytest
import torch

from olmo.config import CustomDatasetCollatorConfig, CustomDatasetConfig
from olmo.data import build_collator
from olmo.data.collator import CustomDatasetDataCollator, DataCollator, PaddingDirection
from olmo.data.util import get_document_lengths


@pytest.mark.parametrize(
    "pad_direction",
    [pytest.param(PaddingDirection.right, id="pad-right"), pytest.param(PaddingDirection.left, id="pad-left")],
)
def test_collate_with_input_ids_tensor(train_config, pad_direction):
    train_config.data.pad_direction = pad_direction
    collator = DataCollator.from_train_config(train_config)

    inputs = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6])]
    batch = collator(inputs)
    assert batch["input_ids"].shape == (2, 4)
    if pad_direction == "right":
        assert batch["input_ids"][1][-1] == train_config.model.pad_token_id
    else:
        assert batch["input_ids"][1][0] == train_config.model.pad_token_id


@pytest.mark.parametrize(
    "pad_direction",
    [pytest.param(PaddingDirection.right, id="pad-right"), pytest.param(PaddingDirection.left, id="pad-left")],
)
def test_collate_with_batch_dict(train_config, pad_direction):
    train_config.data.pad_direction = pad_direction
    collator = DataCollator.from_train_config(train_config)

    inputs = [
        {"input_ids": torch.tensor([0, 1, 2, 3]), "attention_mask": torch.tensor([1, 1, 1, 1])},
        {"input_ids": torch.tensor([4, 5, 6]), "attention_mask": torch.tensor([1, 1, 1])},
    ]
    batch = collator(inputs)  # type: ignore
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"] is not None
    assert batch["attention_mask"].shape == (2, 4)
    if pad_direction == "right":
        assert batch["input_ids"][1][-1] == train_config.model.pad_token_id
        assert batch["attention_mask"][1][-1] == 0
    else:
        assert batch["input_ids"][1][0] == train_config.model.pad_token_id
        assert batch["attention_mask"][1][0] == 0


@pytest.mark.parametrize(
    "pad_direction",
    [pytest.param(PaddingDirection.right, id="pad-right"), pytest.param(PaddingDirection.left, id="pad-left")],
)
def test_collate_with_attention_bias(train_config, pad_direction):
    train_config.data.pad_direction = pad_direction
    collator = DataCollator.from_train_config(train_config)

    inputs = [
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "attention_bias": ~torch.triu(torch.ones(4, 4, dtype=torch.bool)),
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "attention_bias": ~torch.triu(torch.ones(3, 3, dtype=torch.bool)),
        },
    ]
    batch = collator(inputs)  # type: ignore
    assert batch["attention_bias"] is not None
    assert batch["attention_bias"].shape == (2, 1, 4, 4)
    if pad_direction == "right":
        assert (
            batch["attention_bias"][1][0]
            == torch.tensor(
                [
                    [False, False, False, False],
                    [True, False, False, False],
                    [True, True, False, False],
                    [False, False, False, False],
                ]
            )
        ).all()
    else:
        assert (
            batch["attention_bias"][1][0]
            == torch.tensor(
                [
                    [False, False, False, False],
                    [False, False, False, False],
                    [False, True, False, False],
                    [False, True, True, False],
                ]
            )
        ).all()


@pytest.mark.parametrize(
    "pad_direction",
    [pytest.param(PaddingDirection.right, id="pad-right"), pytest.param(PaddingDirection.left, id="pad-left")],
)
def test_collate_with_label_mask(train_config, pad_direction):
    train_config.data.pad_direction = pad_direction
    collator = DataCollator.from_train_config(train_config)

    inputs = [
        {
            "input_ids": torch.tensor([0, 1, 2, 3]),
            "label_mask": torch.tensor([True, False, True, True]),
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "label_mask": torch.tensor([True, True, False]),
        },
    ]
    batch = collator(inputs)  # type: ignore
    assert batch["label_mask"] is not None
    assert batch["label_mask"].shape == (2, 4)
    if pad_direction == "right":
        assert (
            batch["label_mask"]
            == torch.tensor(
                [[True, False, True, True], [True, True, False, False]],
            )
        ).all()
    else:
        assert (
            batch["label_mask"]
            == torch.tensor(
                [[True, False, True, True], [False, True, True, False]],
            )
        ).all()


def test_collate_with_document_lengths(train_config):
    eos_token_id = 50279
    train_config.model.eos_token_id = eos_token_id
    train_config.data.generate_doc_lengths = True
    collator = DataCollator.from_train_config(train_config)

    input_ids = [
        torch.tensor([eos_token_id, 3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5]),
        torch.tensor([3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5, eos_token_id]),
    ]
    inputs = [{"input_ids": x, "doc_lens": get_document_lengths(x, eos_token_id)} for x in input_ids]
    batch = collator(inputs)  # type: ignore
    assert "doc_lens" in batch
    assert "max_doc_lens" in batch
    assert batch["doc_lens"].tolist() == [
        [1, 5, 3, 2],
        [5, 3, 3, 0],
    ]
    assert batch["max_doc_lens"] == [5, 5]


def test_build_collator_default(train_config):
    train_config.data.custom_dataset = None
    collator = build_collator(train_config)
    assert isinstance(collator, DataCollator)
    assert not isinstance(collator, CustomDatasetDataCollator)


def test_build_collator_custom_function(train_config):
    import random

    train_config.data.custom_dataset = CustomDatasetConfig(name="foo", collate_fn="random.uniform")
    collator = build_collator(train_config)
    assert collator is random.uniform


def test_build_custom_collator_with_overrides(train_config):
    eos_token_id = 50279
    train_config.model.eos_token_id = eos_token_id
    train_config.data.generate_doc_lengths = True
    train_config.data.custom_dataset = CustomDatasetConfig(
        name="foo", collate_config=CustomDatasetCollatorConfig(input_id_field="foo", doc_lens_field="bar")
    )
    collator = build_collator(train_config)

    input_ids = [
        torch.tensor([eos_token_id, 3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5]),
        torch.tensor([3, 4, 5, 5, eos_token_id, 6, 5, eos_token_id, 3, 5, eos_token_id]),
    ]
    inputs = [{"foo": x, "bar": get_document_lengths(x, eos_token_id)} for x in input_ids]
    batch = collator(inputs)  # type: ignore
    assert "doc_lens" in batch
    assert "max_doc_lens" in batch
    assert batch["doc_lens"].tolist() == [
        [1, 5, 3, 2],
        [5, 3, 3, 0],
    ]
    assert batch["max_doc_lens"] == [5, 5]
