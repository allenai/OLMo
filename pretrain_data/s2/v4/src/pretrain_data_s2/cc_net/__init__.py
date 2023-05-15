from typing import Any, List, Sequence, Union, overload

from cached_path import cached_path

from .perplexity import DocLM, MultiSentencePiece

# dl_lm:
# 	# Download a pretrained language model
# 	mkdir -p data/lm_sp
# 	wget -c  -P data/lm_sp http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).arpa.bin
# wget -c  -P data/lm_sp
# http://dl.fbaipublicfiles.com/cc_net/lm/$(lang).sp.model

# lm: data/lm_sp/$(lang).sp.model data/lm_sp/$(lang).arpa.bin


class CCNet:
    def get_sp_url(self, lang: str) -> str:
        return f"http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.sp.model"

    def get_lm_url(self, lang: str) -> str:
        return f"http://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin"

    def get_lm_languages(self) -> List[str]:
        langs_str = (
            "af,ar,az,be,bg,bn,ca,cs,da,de,el,en,es,et,fa,fi,fr,gu,he,hi,hr,hu,hy,id,"
            "is,it,ja,ka,kk,km,kn,ko,lt,lv,mk,ml,mn,mr,my,ne,nl,no,pl,pt,ro,ru,uk,zh"
        )
        return langs_str.split(",")

    def __init__(
        self,
        lang: str = "en",
        language_field: str = "language",
        sp_input_field: str = "raw_content",
        sp_output_field: str = "tokenized",
        lm_input_field: str = "tokenized",
        lm_output_field: str = "perplexity",
    ):
        self.input_field = sp_input_field
        self.output_field = lm_output_field
        self.language_field = language_field
        self.lang = lang

        self.sp = MultiSentencePiece(
            models={self.lang: cached_path(self.get_sp_url(self.lang))},
            field=sp_input_field,
            output_field=sp_output_field,
        )
        self.lm = DocLM(
            models={self.lang: cached_path(self.get_lm_url(self.lang))},
            field=lm_input_field,
            output_field=lm_output_field,
            normalize=False,
        )

    @classmethod
    def prefetch(cls):
        """download models by creating a class, then destroying it immediately."""
        (model := cls())(["hello world"])
        del model

    @overload
    def __call__(self, seq: Sequence[str]) -> List[str]:
        ...

    @overload
    def __call__(self, seq: Sequence[dict]) -> List[dict]:
        ...

    def __call__(self, seq: Union[Sequence[str], Sequence[dict]]) -> Union[List[str], List[dict]]:
        seq = [
            {self.input_field: elem, self.language_field: self.lang}
            if (must_extract := isinstance(elem, str))
            else {**elem, self.language_field: self.lang}
            for elem in seq
        ]
        tokenized = self.sp(seq)
        # make sure all output fields are present
        perplexity = [{self.output_field: 0.0, **d} for d in self.lm(tokenized)]

        if must_extract:
            return [str(elem[self.output_field]) for elem in perplexity]

        return perplexity
