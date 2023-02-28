from typing import Dict, List

import pytest
import torch
from cached_path import cached_path

from dolma.config import Config
from dolma.tokenizer import Tokenizer

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
def config() -> Config:
    return Config(vocab_size=50257, eos_token_id=50256, pad_token_id=50256)


@pytest.fixture(scope="function")
def tokenizer(config) -> Tokenizer:
    return Tokenizer.from_pretrained(TEST_MODEL, config)


@pytest.fixture(scope="module")
def eos_token_id(tokenizer: Tokenizer) -> int:
    return tokenizer.eos_token_id


@pytest.fixture(scope="module")
def lorem_ipsum() -> str:
    return LOREM_IPSUM_1.replace("\n", " ").strip()


@pytest.fixture(scope="module")
def lorem_ipsum_docs() -> List[str]:
    return [text.replace("\n", " ").strip() for text in (LOREM_IPSUM_1, LOREM_IPSUM_2)]


@pytest.fixture(scope="module")
def state_dict() -> Dict[str, torch.Tensor]:
    weights_path = cached_path(f"hf://{TEST_MODEL}/pytorch_model.bin")
    with open(weights_path, "rb") as f:
        hf_state_dict = torch.load(f, map_location="cpu")

    def map_key(k: str) -> str:
        if k != "lm_head.weight" and not k.startswith("transformer."):
            k = "transformer." + k
        if k.startswith("transformer.h."):
            k = k.replace("transformer.h.", "transformer.blocks.")
        return k

    def map_val(k: str, v: torch.Tensor) -> torch.Tensor:
        if any(
            k.endswith(s)
            for s in {".attn.c_attn.weight", ".attn.c_proj.weight", ".mlp.c_fc.weight", ".mlp.c_proj.weight"}
        ):
            return v.T
        return v

    state_dict = {
        map_key(k): map_val(k, v)
        for k, v in hf_state_dict.items()
        if not (
            k.endswith(".attn.masked_bias")
            or k.endswith(".attn.bias")
            or k in {"score.weight", "classifier.weight", "classifier.bias"}
        )
    }

    if "lm_head.weight" not in state_dict:
        state_dict["lm_head.weight"] = state_dict["transformer.wte.weight"]

    return state_dict
