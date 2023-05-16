import random
from multiprocessing import current_process

from .base import BaseTagger, TaggerRegistry


@TaggerRegistry.add
class sample(BaseTagger):
    def __init__(self, seed: int = 1) -> None:
        assert seed > 0
        self.seed = ((current_process().pid or 0) + 1) * 1
        random.seed(self.seed)

    def tag(self, text: str) -> dict:
        return {'sample': random.random()}
