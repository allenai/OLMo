"""

Base implementation for a fasttext tagger; all fasttext taggers should inherit from this class.

@kylel, @soldni

"""
import os
from tempfile import NamedTemporaryFile
from typing import Iterable, Literal, NamedTuple, Optional

from cached_path import cached_path
from fasttext import train_supervised
from fasttext.FastText import _FastText
from smashed.utils.io_utils import open_file_for_write

from .data_types import DocResult, Document, Span, TextSlice
from .taggers import BaseTagger
from .utils import split_paragraphs, split_sentences


class Prediction(NamedTuple):
    label: str
    score: float


class BaseFastTextTagger(BaseTagger):
    SENTENCE_LEVEL_TAGGER = "sentence"
    PARAGRAPH_LEVEL_TAGGER = "paragraph"
    DOCUMENT_LEVEL_TAGGER = "document"

    def __init__(self, model_path: str, model_mode: str) -> None:
        # we use this private attribute to avoid a warning from the fasttext library. See this comment:
        # https://github.com/facebookresearch/fastText/issues/1056#issuecomment-1278058705
        self.classifier = _FastText(str(cached_path(model_path)))
        self.mode = model_mode

    @classmethod
    def train(
        cls,
        train_file: str,
        save_path: str,
        learning_rate: float = 0.1,
        word_vectors_dim: int = 100,
        context_window_size: int = 5,
        max_epochs: int = 100,  # non-default
        min_word_count: int = 1,
        min_label_count: int = 1,
        min_char_ngram: int = 0,
        max_char_ngram: int = 0,
        num_negative_samples: int = 5,
        max_word_ngram: int = 2,  # non-default
        loss_function: Literal["ns", "hs", "softmax", "ova"] = "softmax",
        num_buckets: int = 2_000_000,
        num_threads: int = 0,
        learning_rate_update_rate: int = 100,
        sampling_threshold: float = 0.0001,
        label_prefix: str = "__label__",
        verbose: int = 2,
        pretrained_vectors: Optional[str] = None,
    ) -> _FastText:
        # download potentially remote files
        local_train_file = cached_path(train_file)
        local_pretrained_vectors = cached_path(pretrained_vectors) if pretrained_vectors else None

        # base checks on file format
        with open(local_train_file, "r") as f:
            # check a few lines to see if the file is in the right format
            i = 0
            for ln in f:
                if label_prefix not in ln:
                    raise ValueError(f"{train_file} not the fasttext format, no labels found!")
                if (i := i + 1) > 5:
                    break
            if i == 0:
                raise ValueError(f"{train_file} is empty!")

        # train the fasttext model
        classifier = train_supervised(
            input=local_train_file,
            lr=learning_rate,
            dim=word_vectors_dim,
            ws=context_window_size,
            epoch=max_epochs,
            minCount=min_word_count,
            minCountLabel=min_label_count,
            minn=min_char_ngram,
            maxn=max_char_ngram,
            neg=num_negative_samples,
            wordNgrams=max_word_ngram,
            loss=loss_function,
            bucket=num_buckets,
            thread=num_threads,
            lrUpdateRate=learning_rate_update_rate,
            t=sampling_threshold,
            label=label_prefix,
            verbose=verbose,
            pretrainedVectors=local_pretrained_vectors,
        )

        local_save_path = None
        try:
            # create a local temp file where we save the model
            with NamedTemporaryFile("w", delete=False) as f:
                local_save_path = f.name

            # save the model
            classifier.save_model(local_save_path)

            # upload to remote if save_path is s3 path
            with open_file_for_write(save_path, "wb") as fo, open(local_save_path, "rb") as fi:
                fo.write(fi.read())
        finally:
            # regardless to what happened, remove the local temp file if it
            # exists
            if local_save_path is not None and os.path.exists(local_save_path):
                os.remove(local_save_path)

        return classifier

    @classmethod
    def test(
        cls,
        test_file: str,
        model_path: Optional[str] = None,
        classifier: Optional[_FastText] = None,
    ):
        # load the model if one is not provided
        if classifier is None:
            assert model_path is not None, "Please provide either a model path or a model"
            classifier = _FastText(str(cached_path(model_path)))

        local_test_file = cached_path(test_file)
        model_performance = classifier.test(local_test_file)
        print(model_performance)

    def predict(self, doc: Document) -> DocResult:
        if self.mode == self.SENTENCE_LEVEL_TAGGER:
            units = split_sentences(doc.text)
        elif self.mode == self.PARAGRAPH_LEVEL_TAGGER:
            units = split_paragraphs(doc.text)
        elif self.mode == self.DOCUMENT_LEVEL_TAGGER:
            units = [TextSlice(doc=doc.text, start=0, end=len(doc.text))]
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        spans = []
        for unit in units:
            for prediction in self.predict_slice(unit):
                spans.append(Span(start=unit.start, end=unit.end, type=prediction.label, score=prediction.score))

        return DocResult(doc=doc, spans=spans)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        raise NotImplementedError("Please implement the predict slice method")
