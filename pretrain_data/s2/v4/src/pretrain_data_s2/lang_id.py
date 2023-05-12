
from typing import Tuple
from cached_path import cached_path
import fasttext
import pycld2 as cld2


from .consts import FASTTEXT_PATH


class BaseLangId:
    def predict(self, text: str) -> Tuple[str, float]:
        raise NotImplementedError


class FasttextLangId(BaseLangId):
    def __init__(self):
        self.model = fasttext.load_model(str(cached_path(FASTTEXT_PATH)))

    def predict(self, text: str) -> Tuple[str, float]:
        pred = self.model.predict(text.lower().replace("\n", " "))
        lang = pred[0][0].split("__")[-1]   # pyright: ignore
        score = float(pred[1])
        return lang, score


class Cld2LangId(BaseLangId):
    def predict(self, text: str) -> Tuple[str, float]:
        pred = cld2.detect(text)
        lang = pred[2][0][1]
        score = pred[2][0][2]
        return lang, score
