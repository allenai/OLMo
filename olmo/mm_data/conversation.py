from typing import List
from dataclasses import dataclass

DEFAULT_IMAGE_TOKEN = "<image>"

@dataclass
class Conversation:
    """Defines some constants for the conversation dataset"""
    system: str
    roles: List[str]
    offset: int
    role_sep: str = "\n"
    sep: str = "###"
    sep2: str = None


conv_plain = Conversation(
    system="",
    roles=["", ""],
    offset=0,
    sep="", # tokenizer's eos_token
)


conv_olmo_instruct = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["<|user|>", "<|assistant|>"],
    offset=0,
    role_sep="\n",
    sep="\n",
    sep2="<|endoftext|>",
)


conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    offset=0,
    role_sep=": ",
    sep=" ",
    sep2="</s>",
)


conv_templates = {
    "plain": conv_plain,
    "olmo_instruct": conv_olmo_instruct,
    "vicuna_v1": conv_vicuna_v1,
}