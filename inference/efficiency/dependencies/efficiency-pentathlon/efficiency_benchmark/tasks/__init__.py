from typing import Dict, Optional

import datasets
from efficiency_benchmark.task import (
    BINARY_CLASSIFICATION_METRICS,
    ENTAILMENT_METRICS,
    MT_METRICS,
    PERPLEXITY_METRICS,
    QA_METRICS,
    InstanceFormat,
    Task,
    classification_metrics,
    mc_metrics,
)
from efficiency_benchmark.tasks.efficiency_benchmark import (
    EfficiencyBenchmarkPromptTask,
    EfficiencyBenchmarkRaftTask,
    EfficiencyBenchmarkTask,
    efficiency_benchmark_classification_conversion,
    efficiency_benchmark_mt_conversion,
    efficiency_benchmark_prompt_conversion,
    efficiency_benchmark_raft_conversion,
)
from efficiency_benchmark.tasks.eleuther import (
    EleutherClassificationTask,
    EleutherClassificationTaskWithRenamedSplits,
    EleutherTask,
    EleutherTaskWithRenamedSplits,
    RaceEleutherTask,
)
from efficiency_benchmark.tasks.huggingface import (
    HFDatasetsTask,
    hfclassification_conversion,
    hfmc_conversion,
    hfqa_conversion,
)
from efficiency_benchmark.tasks.metaicl import MetaICLTask
from efficiency_benchmark.tasks.mrqa import MrqaTask
from efficiency_benchmark.tasks.p3 import P3Task
from efficiency_benchmark.tasks.t5 import t5_prompt_conversion

TASKS: Dict[str, Task] = {
    "wmt16-en-ro": EfficiencyBenchmarkTask("wmt16", "ro-en")
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_mt_conversion(input_field="en", target_field="ro"),
    )
    .add_metrics(MT_METRICS),  # TODO
    "wmt16-ro-en": EfficiencyBenchmarkTask("wmt16", "ro-en")
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_mt_conversion(input_field="ro", target_field="en"),
    )
    .add_metrics(MT_METRICS),  # TODO
    "wmt14-de-en": EfficiencyBenchmarkTask("wmt14", "de-en")
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_mt_conversion(input_field="de", target_field="en"),
    )
    .add_metrics(MT_METRICS),  # TODO
    "wmt14-en-de": EfficiencyBenchmarkTask("wmt14", "de-en")
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_mt_conversion(input_field="en", target_field="de"),
    )
    .add_metrics(MT_METRICS),
    "wikitext-prompt": EfficiencyBenchmarkPromptTask("wikitext", "wikitext-103-raw-v1").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_prompt_conversion(max_length=128)
    ),
    # RAFT
    "raft::ade_corpus_v2": EfficiencyBenchmarkRaftTask("ade_corpus_v2").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="ade_corpus_v2")
    ),
    "raft::banking_77": EfficiencyBenchmarkRaftTask("banking_77").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="banking_77")
    ),
    "raft::neurips_impact_statement_risks": EfficiencyBenchmarkRaftTask(
        "neurips_impact_statement_risks"
    ).add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_raft_conversion(task_name="neurips_impact_statement_risks"),
    ),
    "raft::one_stop_english": EfficiencyBenchmarkRaftTask("one_stop_english").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="one_stop_english")
    ),
    "raft::overruling": EfficiencyBenchmarkRaftTask("overruling").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="overruling")
    ),
    "raft::semiconductor_org_types": EfficiencyBenchmarkRaftTask(
        "semiconductor_org_types"
    ).add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_raft_conversion(task_name="semiconductor_org_types"),
    ),
    "raft::systematic_review_inclusion": EfficiencyBenchmarkRaftTask(
        "systematic_review_inclusion"
    ).add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_raft_conversion(task_name="systematic_review_inclusion"),
    ),
    "raft::tai_safety_research": EfficiencyBenchmarkRaftTask("tai_safety_research").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="tai_safety_research")
    ),
    "raft::terms_of_service": EfficiencyBenchmarkRaftTask("terms_of_service").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="terms_of_service")
    ),
    "raft::tweet_eval_hate": EfficiencyBenchmarkRaftTask("tweet_eval_hate").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="tweet_eval_hate")
    ),
    "raft::twitter_complaints": EfficiencyBenchmarkRaftTask("twitter_complaints").add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK, efficiency_benchmark_raft_conversion(task_name="twitter_complaints")
    ),
    # from catwalk
    "wikitext": EleutherTask("wikitext").add_metrics(PERPLEXITY_METRICS),
    "piqa": EleutherTask("piqa", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="goal",
            answer_choices_fields=["sol1", "sol2"],
            correct_answer_index_field="label",
        ),
    )
    .add_metrics(mc_metrics(2)),
    "squad": HFDatasetsTask("squad")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "squadshifts-reddit": HFDatasetsTask("squadshifts", "reddit")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "squadshifts-amazon": HFDatasetsTask("squadshifts", "amazon")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "squadshifts-nyt": HFDatasetsTask("squadshifts", "nyt")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "squadshifts-new-wiki": HFDatasetsTask("squadshifts", "new_wiki")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::race": MrqaTask("mrqa", "race")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::newsqa": MrqaTask("mrqa", "newsqa")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::triviaqa": MrqaTask("mrqa", "triviaqa-web")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::searchqa": MrqaTask("mrqa", "searchqa")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::hotpotqa": MrqaTask("mrqa", "hotpotqa")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::naturalquestions": MrqaTask("mrqa", "naturalquestionsshort")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::bioasq": MrqaTask("mrqa", "bioasq")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::drop": MrqaTask("mrqa", "drop")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::relationextraction": MrqaTask("mrqa", "relationextraction")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::textbookqa": MrqaTask("mrqa", "textbookqa")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "mrqa::duorc.paraphraserc": MrqaTask("mrqa", "duorc.paraphraserc")
    .add_instance_conversion(InstanceFormat.HF_QA, hfqa_conversion())
    .add_metrics(QA_METRICS),
    "squad2": EleutherTask("squad2").add_metrics(QA_METRICS),
    "rte": EleutherClassificationTask("rte", answer_options=["True", "False"])
    .add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            use_fields=["sentence1", "sentence2"],
        ),
    )
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="sentence1",
            hypothesis_field="sentence2",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="rte",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="sentence1",
            hypothesis_field="sentence2",
        ),
    ),
    "superglue::rte": HFDatasetsTask("super_glue", "rte")
    .add_instance_conversion(
        InstanceFormat.T5_PROMPT,
        t5_prompt_conversion(
            task_name="rte", label_map={0: "entailment", 1: "not_entailment"}, use_fields=["premise", "hypothesis"]
        ),
    )
    .add_metrics(ENTAILMENT_METRICS),
    "cola": EleutherClassificationTask("cola", answer_options=["no", "yes"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="cola",
            label_map={0: "unacceptable", 1: "acceptable"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field="idx",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="cola",
            label_map={0: "unacceptable", 1: "acceptable"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field="idx",
        ),
    ),
    "mnli": EleutherClassificationTaskWithRenamedSplits("mnli", answer_options=["True", "Neither", "False"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mnli", label_map={0: "entailment", 1: "neutral", 2: "contradiction"}, id_field="idx"
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="mnli", label_map={0: "entailment", 1: "neutral", 2: "contradiction"}, id_field="idx"
        ),
    ),
    "mnli_mismatched": EleutherClassificationTask("mnli_mismatched", answer_options=["True", "Neither", "False"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mnli", label_map={0: "entailment", 1: "neutral", 2: "contradiction"}, id_field="idx"
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="mnli", label_map={0: "entailment", 1: "neutral", 2: "contradiction"}, id_field="idx"
        ),
    ),
    "mrpc": EleutherClassificationTask("mrpc", answer_options=["no", "yes"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="mrpc",
            label_map={0: "not_equivalent", 1: "equivalent"},
            premise_field="sentence1",
            hypothesis_field="sentence2",
            id_field="idx",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="mrpc",
            label_map={0: "not_equivalent", 1: "equivalent"},
            premise_field="sentence1",
            hypothesis_field="sentence2",
            id_field="idx",
        ),
    ),
    "qnli": EleutherClassificationTask("qnli", answer_options=["yes", "no"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="qnli",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="question",
            hypothesis_field="sentence",
            id_field="idx",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="qnli",
            label_map={0: "entailment", 1: "not_entailment"},
            premise_field="question",
            hypothesis_field="sentence",
            id_field="idx",
        ),
    ),
    "qqp": EleutherClassificationTask("qqp", answer_options=["no", "yes"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="qqp",
            label_map={0: "not_duplicate", 1: "duplicate"},
            premise_field="question1",
            hypothesis_field="question2",
            id_field="idx",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="qqp",
            label_map={0: "not_duplicate", 1: "duplicate"},
            premise_field="question1",
            hypothesis_field="question2",
            id_field="idx",
        ),
    ),
    "sst": EleutherClassificationTask("sst", answer_options=["negative", "positive"])
    .add_instance_conversion(
        InstanceFormat.HF_CLASSIFICATION,
        hfclassification_conversion(
            task_name="sst",
            label_map={0: "negative", 1: "positive"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field="idx",
        ),
    )
    .add_instance_conversion(
        InstanceFormat.EFFICIENCY_BENCHMARK,
        efficiency_benchmark_classification_conversion(
            task_name="sst",
            label_map={0: "negative", 1: "positive"},
            premise_field="sentence",
            hypothesis_field=None,
            id_field="idx",
        ),
    ),
    "wnli": EleutherTask("wnli", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "boolq": EleutherTask("boolq", ranked_classification=True).add_metrics(classification_metrics(2)),
    "cb": EleutherTask("cb", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "copa": EleutherTask("copa", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="premise",
            answer_choices_fields=["choice1", "choice2"],
            correct_answer_index_field="label",
            id_field="idx",
        ),
    )
    .add_metrics(mc_metrics(2)),
    "multirc": EleutherTask("multirc", ranked_classification=True).add_metrics(QA_METRICS),
    # "record": EleutherTask("record"),    # record doesn't have a 1:1 correspondence between HF instances and EAI instances
    "wic": EleutherTask("wic", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "wsc": EleutherTask("wsc", ranked_classification=True).add_metrics(mc_metrics(2)),
    # "coqa": EleutherTask("coqa"),  # currently broken in the datasets library
    "drop": EleutherTask("drop").add_metrics(QA_METRICS),
    "lambada": EleutherTask("lambada_standard"),
    "lambada_cloze": EleutherTask("lambada_standard_cloze"),
    "lambada_mt_en": EleutherTask("lambada_openai_mt_en"),
    "lambada_mt_fr": EleutherTask("lambada_openai_mt_fr"),
    "lambada_mt_de": EleutherTask("lambada_openai_mt_de"),
    "lambada_mt_it": EleutherTask("lambada_openai_mt_it"),
    "lambada_mt_es": EleutherTask("lambada_openai_mt_es"),
    "prost": EleutherTask("prost", ranked_classification=True).add_metrics(mc_metrics(4)),
    "mc_taco": EleutherTask("mc_taco", ranked_classification=True).add_metrics(BINARY_CLASSIFICATION_METRICS),
    "pubmedqa": EleutherTaskWithRenamedSplits("pubmedqa").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "sciq": EleutherTask("sciq", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields=["correct_answer", "distractor1", "distractor2", "distractor3"],
            correct_answer_field="correct_answer",
        ),
    )
    .add_metrics(mc_metrics(4)),
    "qa4mre_2011": EleutherTask("qa4mre_2011", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
        ),
    )
    .add_metrics(mc_metrics(5)),
    "qa4mre_2012": EleutherTask("qa4mre_2012", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
        ),
    )
    .add_metrics(mc_metrics(5)),
    "qa4mre_2013": EleutherTask("qa4mre_2013", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="document_str",
            question_field="question_str",
            answer_choices_fields="answer_options.answer_str",
            correct_answer_index_field="correct_answer_id",
            answer_mappings={"1": 0, "2": 1, "3": 2, "4": 3, "5": 4},
        ),
    )
    .add_metrics(mc_metrics(5)),
    "triviaqa": EleutherTask("triviaqa").add_metrics(QA_METRICS),
    "arc_easy": EleutherTask("arc_easy", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id",
            answer_mappings={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3},
        ),
    )
    .add_metrics(mc_metrics(4)),
    "arc_challenge": EleutherTask("arc_challenge", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id",
            answer_mappings={"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "1": 0, "2": 1, "3": 2, "4": 3},
        ),
    )
    .add_metrics(mc_metrics(4)),
    "logiqa": EleutherTask("logiqa", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field="context",
            question_field="question",
            answer_choices_fields="options",
            correct_answer_index_field="label",
        ),
    )
    .add_metrics(mc_metrics(4)),
    "hellaswag": EleutherTask("hellaswag", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="ctx",
            answer_choices_fields="endings",
            correct_answer_index_field="label",
            answer_mappings={"0": 0, "1": 1, "2": 2, "3": 3},
        ),
    )
    .add_metrics(mc_metrics(4)),
    "openbookqa": EleutherTask("openbookqa", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="question_stem",
            answer_choices_fields="choices.text",
            correct_answer_index_field="answerKey",
            id_field="id",
        ),
    )
    .add_metrics(mc_metrics(4)),
    "race": HFDatasetsTask("race", "high").add_metrics(mc_metrics(4)),
    "eleuther::race": RaceEleutherTask().add_metrics(mc_metrics(4)),
    "headqa_es": EleutherTask("headqa_es", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="qtext",
            answer_choices_fields=[
                "answers.0.atext",
                "answers.1.atext",
                "answers.2.atext",
                "answers.3.atext",
                "answers.4.atext",
            ],
            correct_answer_index_field="ra",
            answer_mappings={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        ),
    )
    .add_metrics(mc_metrics(5)),
    "headqa_en": EleutherTask("headqa_en", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="qtext",
            answer_choices_fields=[
                "answers.0.atext",
                "answers.1.atext",
                "answers.2.atext",
                "answers.3.atext",
                "answers.4.atext",
            ],
            correct_answer_index_field="ra",
            answer_mappings={1: 0, 2: 1, 3: 2, 4: 3, 5: 4},
        ),
    )
    .add_metrics(mc_metrics(5)),
    "mathqa": EleutherTask("mathqa", ranked_classification=True).add_metrics(mc_metrics(5)),
    "webqs": EleutherTask("webqs").add_metrics(QA_METRICS),
    "wsc273": EleutherTask("wsc273", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "winogrande": EleutherTask("winogrande", ranked_classification=True)
    .add_instance_conversion(
        InstanceFormat.HF_MC,
        hfmc_conversion(
            context_field=None,
            question_field="sentence",
            answer_choices_fields=["option1", "option2"],
            correct_answer_index_field="answer",
            answer_mappings={"1": 0, "2": 1},
        ),
    )
    .add_metrics(mc_metrics(2)),
    "anli_r1": EleutherTask("anli_r1", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "anli_r2": EleutherTask("anli_r2", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "anli_r3": EleutherTask("anli_r3", ranked_classification=True).add_metrics(ENTAILMENT_METRICS),
    "ethics_cm": EleutherTask("ethics_cm").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_deontology": EleutherTask("ethics_deontology").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_justice": EleutherTask("ethics_justice").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_utilitarianism_original": EleutherTask("ethics_utilitarianism_original").add_metrics(
        BINARY_CLASSIFICATION_METRICS
    ),
    "ethics_utilitarianism": EleutherTask("ethics_utilitarianism").add_metrics(BINARY_CLASSIFICATION_METRICS),
    "ethics_virtue": EleutherTask("ethics_virtue").add_metrics(BINARY_CLASSIFICATION_METRICS),
    # "truthfulqa_mc": EleutherTask("truthfulqa_mc", ranked_classification=True),
    "truthfulqa_gen": EleutherTask("truthfulqa_gen"),
    "mutual": EleutherTask("mutual"),
    "mutual_plus": EleutherTask("mutual_plus"),
    "math_algebra": EleutherTask("math_algebra").add_metrics(QA_METRICS),
    "math_counting_and_prob": EleutherTask("math_counting_and_prob").add_metrics(QA_METRICS),
    "math_geometry": EleutherTask("math_geometry").add_metrics(QA_METRICS),
    "math_intermediate_algebra": EleutherTask("math_intermediate_algebra").add_metrics(QA_METRICS),
    "math_num_theory": EleutherTask("math_num_theory").add_metrics(QA_METRICS),
    "math_prealgebra": EleutherTask("math_prealgebra").add_metrics(QA_METRICS),
    "math_precalc": EleutherTask("math_precalc").add_metrics(QA_METRICS),
    "math_asdiv": EleutherTask("math_asdiv").add_metrics(QA_METRICS),
    "arithmetic_2da": EleutherTask("arithmetic_2da").add_metrics(QA_METRICS),
    "arithmetic_2ds": EleutherTask("arithmetic_2ds").add_metrics(QA_METRICS),
    "arithmetic_3da": EleutherTask("arithmetic_3da").add_metrics(QA_METRICS),
    "arithmetic_3ds": EleutherTask("arithmetic_3ds").add_metrics(QA_METRICS),
    "arithmetic_4da": EleutherTask("arithmetic_4da").add_metrics(QA_METRICS),
    "arithmetic_4ds": EleutherTask("arithmetic_4ds").add_metrics(QA_METRICS),
    "arithmetic_5da": EleutherTask("arithmetic_5da").add_metrics(QA_METRICS),
    "arithmetic_5ds": EleutherTask("arithmetic_5ds").add_metrics(QA_METRICS),
    "arithmetic_2dm": EleutherTask("arithmetic_2dm").add_metrics(QA_METRICS),
    "arithmetic_1dc": EleutherTask("arithmetic_1dc").add_metrics(QA_METRICS),
    # "iwslt17-en-ar": EleutherTask("iwslt17-en-ar"),    # no support for translations tasks for now
    # "iwslt17-ar-en": EleutherTask("iwslt17-ar-en"),    # no support for translations tasks for now
    "anagrams1": EleutherTask("anagrams1").add_metrics(QA_METRICS),
    "anagrams2": EleutherTask("anagrams2").add_metrics(QA_METRICS),
    "cycle_letters": EleutherTask("cycle_letters").add_metrics(QA_METRICS),
    "random_insertion": EleutherTask("random_insertion").add_metrics(QA_METRICS),
    "reversed_words": EleutherTask("reversed_words").add_metrics(QA_METRICS),
    # MetaICL
    "metaicl::piqa": MetaICLTask("piqa").add_metrics(mc_metrics(2)),
    "metaicl::boolq": MetaICLTask("boolq").add_metrics(classification_metrics(2)),
    "metaicl::tweet_eval-stance_feminist": MetaICLTask("tweet_eval-stance_feminist").add_metrics(
        classification_metrics(3)
    ),
    "metaicl::ethos-national_origin": MetaICLTask("ethos-national_origin").add_metrics(classification_metrics(2)),
    "metaicl::tweet_eval-hate": MetaICLTask("tweet_eval-hate").add_metrics(classification_metrics(2)),
    "metaicl::ag_news": MetaICLTask("ag_news").add_metrics(classification_metrics(4)),
    "metaicl::amazon_polarity": MetaICLTask("amazon_polarity").add_metrics(classification_metrics(2)),
    "metaicl::hate_speech18": MetaICLTask("hate_speech18").add_metrics(classification_metrics(2)),
    "metaicl::poem_sentiment": MetaICLTask("poem_sentiment").add_metrics(classification_metrics(3)),
    "metaicl::climate_fever": MetaICLTask("climate_fever").add_metrics(classification_metrics(4)),
    "metaicl::medical_questions_pairs": MetaICLTask("medical_questions_pairs").add_metrics(
        classification_metrics(2)
    ),
    "metaicl::tweet_eval-stance_atheism": MetaICLTask("tweet_eval-stance_atheism").add_metrics(
        classification_metrics(3)
    ),
    "metaicl::superglue-cb": MetaICLTask("superglue-cb").add_metrics(classification_metrics(3)),
    "metaicl::dbpedia_14": MetaICLTask("dbpedia_14").add_metrics(classification_metrics(14)),
    "metaicl::wiki_qa": MetaICLTask("wiki_qa").add_metrics(classification_metrics(2)),
    "metaicl::emo": MetaICLTask("emo").add_metrics(classification_metrics(4)),
    "metaicl::yelp_polarity": MetaICLTask("yelp_polarity").add_metrics(classification_metrics(2)),
    "metaicl::ethos-religion": MetaICLTask("ethos-religion").add_metrics(classification_metrics(2)),
    "metaicl::financial_phrasebank": MetaICLTask("financial_phrasebank").add_metrics(classification_metrics(3)),
    "metaicl::tab_fact": MetaICLTask("tab_fact").add_metrics(classification_metrics(2)),
    "metaicl::anli": MetaICLTask("anli").add_metrics(classification_metrics(3)),
    "metaicl::ethos-race": MetaICLTask("ethos-race").add_metrics(classification_metrics(2)),
    "metaicl::glue-mrpc": MetaICLTask("glue-mrpc").add_metrics(classification_metrics(2)),
    "metaicl::glue-qqp": MetaICLTask("glue-qqp").add_metrics(classification_metrics(2)),
    # "metaicl::medical_questions_pairs": MetaICLTask("medical_questions_pairs").add_metrics(classification_metrics(2)),
    "metaicl::paws": MetaICLTask("paws").add_metrics(classification_metrics(2)),
    # "metaicl::anli": MetaICLTask("anli").add_metrics(classification_metrics(3)),
    "metaicl::glue-mnli": MetaICLTask("glue-mnli").add_metrics(classification_metrics(3)),
    "metaicl::glue-qnli": MetaICLTask("glue-qnli").add_metrics(classification_metrics(2)),
    "metaicl::glue-rte": MetaICLTask("glue-rte").add_metrics(classification_metrics(2)),
    "metaicl::glue-wnli": MetaICLTask("glue-wnli").add_metrics(classification_metrics(2)),
    "metaicl::scitail": MetaICLTask("scitail").add_metrics(classification_metrics(2)),
    "metaicl::sick": MetaICLTask("sick").add_metrics(classification_metrics(3)),
    # "metaicl::superglue-cb": MetaICLTask("superglue-cb").add_metrics(classification_metrics(3)),
    "metaicl::ai2_arc": MetaICLTask("ai2_arc").add_metrics(mc_metrics(4)),
    "metaicl::codah": MetaICLTask("codah").add_metrics(mc_metrics(4)),
    "metaicl::cosmos_qa": MetaICLTask("cosmos_qa").add_metrics(mc_metrics(4)),
    "metaicl::dream": MetaICLTask("dream").add_metrics(mc_metrics(3)),
    "metaicl::hellaswag": MetaICLTask("hellaswag").add_metrics(mc_metrics(4)),
    "metaicl::openbookqa": MetaICLTask("openbookqa").add_metrics(mc_metrics(4)),
    "metaicl::qasc": MetaICLTask("qasc").add_metrics(mc_metrics(8)),
    "metaicl::quail": MetaICLTask("quail").add_metrics(mc_metrics(4)),
    "metaicl::quarel": MetaICLTask("quarel").add_metrics(mc_metrics(2)),
    "metaicl::quartz-no_knowledge": MetaICLTask("quartz-no_knowledge").add_metrics(mc_metrics(2)),
    "metaicl::quartz-with_knowledge": MetaICLTask("quartz-with_knowledge").add_metrics(mc_metrics(2)),
    "metaicl::sciq": MetaICLTask("sciq").add_metrics(mc_metrics(4)),
    "metaicl::superglue-copa": MetaICLTask("superglue-copa").add_metrics(mc_metrics(2)),
    "metaicl::swag": MetaICLTask("swag").add_metrics(mc_metrics(4)),
    "metaicl::wino_grande": MetaICLTask("wino_grande").add_metrics(mc_metrics(2)),
    "metaicl::wiqa": MetaICLTask("wiqa").add_metrics(mc_metrics(3)),
    "metaicl::unifiedqa:qasc": MetaICLTask("unifiedqa:qasc").add_metrics(mc_metrics(8)),
    "metaicl::unifiedqa:qasc_with_ir": MetaICLTask("unifiedqa:qasc_with_ir").add_metrics(mc_metrics(8)),
    "metaicl::unifiedqa:openbookqa": MetaICLTask("unifiedqa:openbookqa").add_metrics(mc_metrics(4)),
    "metaicl::unifiedqa:openbookqa_with_ir": MetaICLTask("unifiedqa:openbookqa_with_ir").add_metrics(
        mc_metrics(4)
    ),
    "metaicl::unifiedqa:mctest": MetaICLTask("unifiedqa:mctest").add_metrics(mc_metrics(4)),
    "metaicl::unifiedqa:ai2_science_middle": MetaICLTask("unifiedqa:ai2_science_middle").add_metrics(
        mc_metrics(4)
    ),
    "metaicl::numer_sense": MetaICLTask("numer_sense").add_metrics(classification_metrics(12)),
    "metaicl::race-high": MetaICLTask("race-high").add_metrics(mc_metrics(4)),
    "metaicl::commonsense_qa": MetaICLTask("commonsense_qa").add_metrics(mc_metrics(5)),
}

for config in datasets.get_dataset_config_names("bigscience/P3"):
    TASKS[f"p3::{config}"] = P3Task(config)

TASK_SETS = {
    "iz": {
        "arc_challenge",
        "arc_easy",
        "boolq",
        "copa",
        "headqa_en",
        "hellaswag",
        "lambada",
        "logiqa",
        "mathqa",
        "mc_taco",
        "mrpc",
        "multirc",
        "openbookqa",
        "piqa",
        "prost",
        "pubmedqa",
        "qnli",
        "qqp",
        "race",
        "rte",
        "sciq",
        "sst",
        "triviaqa",
        "webqs",
        "wic",
        "winogrande",
        "wnli",
        "wsc",
    },
    "raft": {name for name in TASKS.keys() if name.startswith("raft::")},
    "metaicl-classification-eval": {
        "metaicl::tweet_eval-stance_feminist",
        "metaicl::ethos-national_origin",
        "metaicl::tweet_eval-hate",
        "metaicl::ag_news",
        "metaicl::amazon_polarity",
        "metaicl::hate_speech18",
        "metaicl::poem_sentiment",
        "metaicl::climate_fever",
        "metaicl::medical_questions_pairs",
        "metaicl::tweet_eval-stance_atheism",
        "metaicl::superglue-cb",
        "metaicl::dbpedia_14",
        "metaicl::wiki_qa",
        "metaicl::emo",
        "metaicl::yelp_polarity",
        "metaicl::ethos-religion",
        "metaicl::financial_phrasebank",
        "metaicl::tab_fact",
        "metaicl::anli",
        "metaicl::ethos-race",
    },
    "metaicl-paraphrase-eval": {
        "metaicl::glue-mrpc",
        "metaicl::glue-qqp",
        "metaicl::medical_questions_pairs",
        "metaicl::paws",
    },
    "metaicl-nli-eval": {
        "metaicl::anli",
        "metaicl::glue-mnli",
        "metaicl::glue-qnli",
        "metaicl::glue-rte",
        "metaicl::glue-wnli",
        "metaicl::scitail",
        "metaicl::sick",
        "metaicl::superglue-cb",
    },
    "metaicl-qa-eval": {
        "metaicl::ai2_arc",
        "metaicl::codah",
        "metaicl::cosmos_qa",
        "metaicl::dream",
        "metaicl::hellaswag",
        "metaicl::openbookqa",
        "metaicl::qasc",
        "metaicl::quail",
        "metaicl::quarel",
        "metaicl::quartz-no_knowledge",
        "metaicl::quartz-with_knowledge",
        "metaicl::sciq",
        "metaicl::superglue-copa",
        "metaicl::swag",
        "metaicl::wino_grande",
        "metaicl::wiqa",
        "metaicl::unifiedqa:qasc",
        "metaicl::unifiedqa:qasc_with_ir",
        "metaicl::unifiedqa:openbookqa",
        "metaicl::unifiedqa:openbookqa_with_ir",
        "metaicl::unifiedqa:mctest",
        "metaicl::unifiedqa:ai2_science_middle",
    },
    "metaicl-lr-eval": {
        "metaicl::quarel",
        "metaicl::financial_phrasebank",
        "metaicl::openbookqa",
        "metaicl::codah",
        "metaicl::qasc",
        "metaicl::glue-mrpc",
        "metaicl::dream",
        "metaicl::sick",
        "metaicl::commonsense_qa",
        "metaicl::medical_questions_pairs",
        "metaicl::quartz-with_knowledge",
        "metaicl::poem_sentiment",
        "metaicl::quartz-no_knowledge",
        "metaicl::glue-wnli",
        "metaicl::climate_fever",
        "metaicl::ethos-national_origin",
        "metaicl::ethos-race",
        "metaicl::ethos-religion",
        "metaicl::ai2_arc",
        "metaicl::hate_speech18",
        "metaicl::glue-rte",
        "metaicl::superglue-cb",
        "metaicl::superglue-copa",
        "metaicl::tweet_eval-hate",
        "metaicl::tweet_eval-stance_atheism",
        "metaicl::tweet_eval-stance_feminist",
    },
}


def short_name_for_task_object(task: Task) -> Optional[str]:
    for task_name, task_object in TASKS.items():
        if id(task) == id(task_object):
            return task_name
    return None
