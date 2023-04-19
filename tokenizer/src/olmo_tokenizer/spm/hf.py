# class SentencePieceUnigramTokenizer(BaseTokenizer):
#     """SentencePiece Unigram Tokenizer

#     Represents the Unigram algorithm, with the pretokenization used by SentencePiece
#     """

#     def __init__(
#         self,
#         vocab: Optional[str] = None,
#         replacement: str = "▁",
#         add_prefix_space: bool = True,
#     ):
#         if vocab is not None:
#             # Let Unigram(..) fail if only one of them is None
#             tokenizer = Tokenizer(Unigram(vocab))
#         else:
#             tokenizer = Tokenizer(Unigram())

#         tokenizer.normalizer = normalizers.Sequence(
#             [normalizers.Nmt(), normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")]
#         )
#         tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)
#         tokenizer.decoder = decoders.Metaspace(replacement=replacement, add_prefix_space=add_prefix_space)

#         parameters = {
#             "model": "SentencePieceUnigram",
#             "replacement": replacement,
#             "add_prefix_space": add_prefix_space,
#         }

#         super().__init__(tokenizer, parameters)
