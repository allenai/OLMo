from typing import List, Tuple, Union, overload

import cld3
import pycld2 as cld2
from cached_path import cached_path
from fasttext.FastText import _FastText

from .consts import FASTTEXT_PATH


class BaseLangId:
    def predict(self, text: str) -> Tuple[str, float]:
        raise NotImplementedError

    @overload
    def get_language(self, texts: str) -> Tuple[str, float]:
        ...

    @overload
    def get_language(self, texts: List[str]) -> List[Tuple[str, float]]:
        ...

    def get_language(
        self,
        texts: Union[List[str], str],
    ) -> Union[List[Tuple[str, float]], Tuple[str, float]]:
        langs: Union[List[Tuple[str, float]], None] = [] if isinstance(texts, list) else None
        texts = [texts] if isinstance(texts, str) else texts

        for text in texts:
            try:
                text = text.strip()
                lang, prob = self.predict(text)
            except Exception:
                lang, prob = "unk", 1.0

            if langs is None:
                return lang, prob

            langs.append((lang, prob))

        assert langs is not None
        return langs

    @overload
    def get_language_dict(self, texts: str) -> dict:
        ...

    @overload
    def get_language_dict(self, texts: List[str]) -> List[dict]:
        ...

    def get_language_dict(self, texts: Union[List[str], str]) -> Union[List[dict], dict]:
        output = self.get_language(texts)
        if isinstance(output, tuple):
            return {"lang": output[0], "prob": output[1]}
        else:
            return [{"lang": lang, "prob": prob} for lang, prob in output]


class FasttextLangId(BaseLangId):
    def __init__(self):
        # we use this private attribute to avoid a warning from the fasttext library
        # see this comment:
        # https://github.com/facebookresearch/fastText/issues/1056#issuecomment-1278058705
        self.model = _FastText(model_path=str(cached_path(FASTTEXT_PATH)))

    def predict(self, text: str) -> Tuple[str, float]:
        pred = self.model.predict(text.lower().replace("\n", " "))
        lang = pred[0][0].split("__")[-1]  # pyright: ignore
        score = float(pred[1])
        return lang, score


class Cld2LangId(BaseLangId):
    def predict(self, text: str) -> Tuple[str, float]:
        pred = cld2.detect(text)
        lang = pred[2][0][1]
        score = pred[2][0][2]
        score /= 100.0  # score is in percentage
        lang = "unk" if lang == "un" else lang
        return lang, score


class Cld3LangId(BaseLangId):
    def predict(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)  # pyright: ignore
        lang = pred.language
        score = pred.probability
        return lang, score
