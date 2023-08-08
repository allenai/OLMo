from typing import List

import pytest

from olmo.config import (
    DataConfig,
    InitFnType,
    ModelConfig,
    OptimizerConfig,
    PaddingDirection,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
)
from olmo.tokenizer import Tokenizer

TEST_MODEL = "gpt2"

LOREM_IPSUM_1 = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit,
sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip
ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit
esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat
non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

LOREM_IPSUM_2 = """
Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque
laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi
architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia
voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores
eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem
ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius
modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem.
Ut enim ad minima veniam, quis nostrum exercitationem ullam corporis suscipit
laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem vel eum iure
reprehenderit qui in ea voluptate velit esse quam nihil molestiae consequatur,
vel illum qui dolorem eum fugiat quo voluptas nulla pariatur?
"""


@pytest.fixture(scope="function")
def model_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=50257,
        eos_token_id=50256,
        pad_token_id=50256,
        d_model=128,
        n_heads=2,
        n_layers=3,
        max_sequence_length=512,
        init_fn=InitFnType.normal,
    )


@pytest.fixture(scope="function")
def tokenizer() -> Tokenizer:
    return Tokenizer.from_pretrained(TEST_MODEL)


@pytest.fixture(scope="function")
def train_config(tmp_path, model_config) -> TrainConfig:
    return TrainConfig(
        model=model_config,
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(),
        data=DataConfig(
            paths=[
                "test_fixtures/c4-sample.01.json.gz",
                "test_fixtures/c4-sample.02.json.gz",
                "test_fixtures/c4-sample.03.json.gz",
            ],
            pad_direction=PaddingDirection.right,
        ),
        tokenizer=TokenizerConfig(identifier=TEST_MODEL),
        save_folder=str(tmp_path / "checkpoints"),
    )


@pytest.fixture(scope="module")
def eos_token_id(tokenizer: Tokenizer) -> int:
    return tokenizer.eos_token_id


@pytest.fixture(scope="module")
def lorem_ipsum() -> str:
    return LOREM_IPSUM_1.replace("\n", " ").strip()


@pytest.fixture(scope="module")
def lorem_ipsum_docs() -> List[str]:
    return [text.replace("\n", " ").strip() for text in (LOREM_IPSUM_1, LOREM_IPSUM_2)]


@pytest.fixture(scope="function")
def model_path() -> str:
    return "test_fixtures/test-olmo-model"
