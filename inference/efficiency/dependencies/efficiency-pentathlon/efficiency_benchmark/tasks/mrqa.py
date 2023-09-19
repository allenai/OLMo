import functools

from datasets import load_dataset
from efficiency_benchmark.tasks import HFDatasetsTask


class MrqaTask(HFDatasetsTask):
    TEST_DATASETS = {"race", "drop", "bioasq", "relationextraction", "textbookqa", "duorc.paraphraserc"}
    DEV_DATASETS = {"newsqa", "searchqa", "triviaqa-web", "naturalquestionsshort", "hotpotqa"}

    @functools.lru_cache
    def has_split(self, split: str) -> bool:
        if self.dataset_name in self.TEST_DATASETS:
            return split == "test"
        elif self.dataset_name in self.DEV_DATASETS:
            return split == "validation"

        return False

    @functools.lru_cache
    def dataset(self, split: str):
        assert self.dataset_name is not None, "MRQA requires a dataset name as it contains multiple subsets"
        assert_message = (
            f"Specified task, {self.dataset_name}, is not in specified split, {split}."
            if split in {"validation", "test"}
            else f"No such split {split}."
        )
        assert self.has_split(split), assert_message

        def filter_subset(example):
            return example["subset"].lower() == self.dataset_name

        loaded_dataset = load_dataset(self.dataset_path, split=split).filter(
            filter_subset, load_from_cache_file=False
        )  # caching results in wrong loading of cached dataset

        # rename columns to match SQuAD format
        loaded_dataset = loaded_dataset.rename_column("qid", "id")
        loaded_dataset = loaded_dataset.remove_columns(["context_tokens", "question_tokens", "answers"])
        loaded_dataset = loaded_dataset.rename_column("detected_answers", "answers")

        # preprocess answers to match format expected by SQuAD metric
        def preprocess_answers(example):
            example["answers"].update({"answer_start": [x["start"][0] for x in example["answers"]["char_spans"]]})
            del example["answers"]["char_spans"]
            del example["answers"]["token_spans"]
            return example

        loaded_dataset = loaded_dataset.map(preprocess_answers, load_from_cache_file=False)
        return loaded_dataset
