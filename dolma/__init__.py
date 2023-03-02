from .config import Config
from .model import DolmaGPT, DolmaGPTOutput
from .tokenizer import Tokenizer, TruncationDirection

__all__ = ["Config", "Tokenizer", "TruncationDirection", "DolmaGPT", "DolmaGPTOutput", "check_install"]


def check_install(cuda: bool = False):
    import torch

    from .version import VERSION

    if cuda:
        assert torch.cuda.is_available(), "CUDA is not available!"
        print("CUDA available")

    print(f"DOLMA v{VERSION} installed")
