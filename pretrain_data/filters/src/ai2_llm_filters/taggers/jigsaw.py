"""

Filters.

@kylel, @soldni

"""

from typing import Iterable

from ..core_tools.data_types import TextSlice
from ..core_tools.ft_tagger import BaseFastTextTagger, Prediction
from ..core_tools.registry import TaggerRegistry


@TaggerRegistry.add("jigsaw_hatespeech_document_v2")
class FastTextJigsawHatespeechDocumentTagger(BaseFastTextTagger):
    MODEL_PATH = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/aakankshan/olmo-data-filters/jigsaw_fasttext_bigrams_hatespeech_final.bin"  # noqa: E501

    def __init__(self):
        super().__init__(model_path=self.MODEL_PATH, model_mode=self.DOCUMENT_LEVEL_TAGGER)

    def predict_slice(self, text_slice: TextSlice) -> Iterable[Prediction]:
        labels, probs = self.classifier.predict(text_slice.text.replace("\n", " ").strip(), k=-1)
        label_index = 1 if "non" in labels[0] else 0  # pyright: ignore
        return (
            Prediction(label=labels[label_index], score=probs[label_index]),
            Prediction(label=labels[1 - label_index], score=probs[1 - label_index]),
        )


@TaggerRegistry.add("jigsaw_hatespeech_sentence_v2")
class FastTextJigsawHatespeechSentenceTagger(FastTextJigsawHatespeechDocumentTagger):
    def __init__(self):
        BaseFastTextTagger.__init__(self, model_path=self.MODEL_PATH, model_mode=self.SENTENCE_LEVEL_TAGGER)


@TaggerRegistry.add("jigsaw_nsfw_document_v1")
class FastTextJigsawNsfwDocumentTagger(FastTextJigsawHatespeechDocumentTagger):
    MODEL_PATH = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/aakankshan/olmo-data-filters/jigsaw_fasttext_bigrams_nsfw_final.bin"  # noqa: E501


@TaggerRegistry.add("jigsaw_nsfw_sencence_v2")
class FastTextJigsawNsfwSentenceTagger(FastTextJigsawHatespeechSentenceTagger):
    MODEL_PATH = "https://ai2-s2-research-public.s3.us-west-2.amazonaws.com/aakankshan/olmo-data-filters/jigsaw_fasttext_bigrams_nsfw_final.bin"  # noqa: E501
