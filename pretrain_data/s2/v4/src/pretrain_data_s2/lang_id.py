from typing import List, Tuple, Union, overload

import cld3
import pycld2 as cld2
from cached_path import cached_path
from fasttext.FastText import _FastText

from .consts import FASTTEXT_PATH, LANG_ID_CUT


class BaseLangId:
    def predict(self, text: str) -> Tuple[str, float]:
        raise NotImplementedError

    @overload
    def get_language(self, texts: str, cutoff: int = LANG_ID_CUT) -> str:
        ...

    @overload
    def get_language(self, texts: List[str], cutoff: int = LANG_ID_CUT) -> List[str]:
        ...

    def get_language(self, texts: Union[List[str], str], cutoff: int = LANG_ID_CUT) -> Union[List[str], str]:
        langs: Union[List[str], None] = [] if isinstance(texts, list) else None
        texts = [texts] if isinstance(texts, str) else texts

        for text in texts:
            try:
                text = text.strip()[:cutoff]
                lang, _ = self.predict(text)
                # langs.append(lang)  # type: ignore
            except Exception:
                lang = "unk"

            if langs is None:
                return lang

            langs.append(lang)

        assert langs is not None
        return langs


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
        lang = "unk" if lang == "un" else lang
        return lang, score


class Cld3LangId(BaseLangId):
    def predict(self, text: str) -> Tuple[str, float]:
        pred = cld3.get_language(text)  # pyright: ignore
        lang = pred.language
        score = pred.probability
        return lang, score
