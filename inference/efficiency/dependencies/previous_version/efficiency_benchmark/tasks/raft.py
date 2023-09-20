from typing import Any, Dict, List, Optional

from efficiency_benchmark.task import (
    InstanceFormat,
    RankClassificationInstance,
    classification_metrics,
)
from efficiency_benchmark.tasks import HFDatasetsTask

_FIELD_ORDERING = {
    "ade_corpus_v2": ["Sentence"],
    "banking_77": ["Query"],
    "terms_of_service": ["Sentence"],
    "tai_safety_research": ["Title", "Abstract Note", "Publication Title", "Item Type", "Publication Year"],
    "neurips_impact_statement_risks": ["Impact statement", "Paper title"],
    "overruling": ["Sentence"],
    "systematic_review_inclusion": ["Title", "Abstract", "Journal"],
    "one_stop_english": ["Article"],
    "tweet_eval_hate": ["Tweet"],
    "twitter_complaints": ["Tweet text"],
    "semiconductor_org_types": ["Organization name", "Paper title"],
}
_INSTRUCTIONS = {
    "ade_corpus_v2": "Label the sentence based on whether it is related to an adverse drug effect (ADE). Details are described below:\nDrugs: Names of drugs and chemicals that include brand names, trivial names, abbreviations and systematic names were annotated. Mentions of drugs or chemicals should strictly be in a therapeutic context. This category does not include the names of metabolites, reaction byproducts, or hospital chemicals (e.g. surgical equipment disinfectants).\nAdverse effect: Mentions of adverse effects include signs, symptoms, diseases, disorders, acquired abnormalities, deficiencies, organ damage or death that strictly occur as a consequence of drug intake.",
    "banking_77": "The following is a banking customer service query. Classify the query into one of the 77 categories available.",
    "terms_of_service": "Label the sentence from a Terms of Service based on whether it is potentially unfair. If it seems clearly unfair, mark it as potentially unfair.\nAccording to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer. \nDetails on types of potentially unfair clauses are found below:\nThe jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses giving consumers a right to bring disputes in their place of residence were marked as clearly fair, whereas clauses stating that any judicial proceeding takes a residence away were marked as clearly unfair.\nThe choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. Clauses defining the applicable law as the law of the consumer's country of residence were marked as clearly fair. In every other case, the choice of law clause was considered as potentially unfair.\nThe limitation of liability clause stipulates that the duty to pay damages is limited or excluded, for certain kind of losses, under certain conditions. Clauses that explicitly affirm non-excludable providers' liabilities were marked as clearly fair. Clauses that reduce, limit, or exclude the liability of the service provider were marked as potentially unfair when concerning broad categories of losses or causes of them.\nThe unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clause was always considered as potentially unfair.\nThe unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.\nThe contract by using clause stipulates that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them. We always marked such clauses as potentially unfair.\nThe content removal gives the provider a right to modify/delete user's content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so.\nThe arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court. Clauses stipulating that the arbitration should take place in a state other then the state of consumer's residence or be based on arbiter's discretion were marked as clearly unfair. Clauses defining arbitration as fully optional were marked as clearly fair.",
    "tai_safety_research": 'Transformative AI (TAI) is defined as AI that precipitates a transition comparable to (or more significant than) the agricultural or industrial revolution. Label a paper as "TAI safety research" if: \n1. The contents of the paper are directly motivated by, and substantively inform, the challenge of ensuring good outcomes for TAI, \n2. There is substantive content on AI safety, not just AI capabilities, \n3. The intended audience is the community of researchers, \n4. It meets a subjective threshold of seriousness/quality, \n5. Peer review is not required.',
    "neurips_impact_statement_risks": "Label the impact statement based on whether it mentions a harmful application of the research done in the paper. Make sure the statement is sufficient to conclude there are harmful applications of the research being done, not a past risk that this research is solving.",
    "overruling": "In law, an overruling sentence is a statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by the same or higher ranking court which establishes a different rule on the point of law involved. Label the sentence based on whether it is overruling or not.",
    "systematic_review_inclusion": "Identify whether this paper should be included in a meta-review which includes the findings of systematic reviews on interventions designed to promote charitable donations. \nIncluded reviews should describe monetary charitable donations, assess any population of participants in any context, and be peer reviewed and written in English. \nThey should not report new data, be non-systematic reviews, consider cause-related marketing or other kinds of prosocial behaviour.",
    "one_stop_english": "The following is an article sourced from The Guardian newspaper, and rewritten by teachers to suit three levels of adult English as Second Language (ESL) learners: elementary, intermediate, and advanced. Predict the level of the article.",
    "tweet_eval_hate": "Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.",
    "twitter_complaints": "A complaint presents a state of affairs which breaches the writer\u2019s favorable expectation. Label the tweet text based on whether it contains a complaint.",
    "semiconductor_org_types": 'The dataset is a list of institutions that have contributed papers to semiconductor conferences in the last 25 years, as catalogued by IEEE and sampled randomly. The goal is to classify the institutions into one of three categories: "university", "company" or "research institute".',
}
assert _FIELD_ORDERING.keys() == _INSTRUCTIONS.keys()


class RaftTask(HFDatasetsTask):
    def __init__(self, subset: str, number_of_classes: int = 2):
        self.subset = subset
        if subset not in _FIELD_ORDERING:
            raise ValueError(f"RAFT subset {subset} not found")
        super().__init__("ought/raft", subset)
        self.add_instance_conversion(InstanceFormat.RANK_CLASSIFICATION, self.instance_as_rank_classification)
        self.add_instance_conversion(InstanceFormat.ELEUTHER_REQUESTS, self.instance_as_eleuther_requests)
        self.add_metrics(classification_metrics(number_of_classes))

    @property
    def _field_ordering(self):
        return _FIELD_ORDERING[self.subset]

    @property
    def instructions(self):
        return _INSTRUCTIONS[self.subset].strip()

    @property
    def answer_choices(self) -> List[str]:
        # Label 0 is "unlabeled"
        result = self.dataset("train").features["Label"].names[1:]
        if self.subset == "banking_77":
            result = [answer.replace("_", " ").replace(". ", " ") for answer in result]
        return result

    @property
    def default_split(self) -> str:
        # RAFT doesn't have labels in the test split
        return "train"

    def instance_as_rank_classification(
        self,
        instance: Dict[str, Any],
        *,
        include_instructions: bool = False,
        include_labels_in_instructions: bool = False,
        fewshot_instances: Optional[List[Dict[str, Any]]] = None,
    ) -> RankClassificationInstance:
        if include_instructions:
            if include_labels_in_instructions:
                prefix = self.instructions + " Possible labels: " + ", ".join(self.answer_choices)
            else:
                prefix = self.instructions
        else:
            prefix = ""

        if fewshot_instances is None:
            fewshot_instances = []
        for fewshot_instance in fewshot_instances:
            as_mc = self.instance_as_rank_classification(fewshot_instance)
            if as_mc.correct_choice is None:
                raise ValueError("Could not determine correct choice in ranked classification instance.")
            correct_choice = as_mc.choices[as_mc.correct_choice]
            prefix += f"{correct_choice[0].strip()} {correct_choice[1].strip()}\n\n"

        tuples = []
        for answer_choice in self.answer_choices:
            input_str = prefix
            for key in self._field_ordering:
                value = instance[key].strip()
                if len(value) > 0:
                    input_str += f" {key}: {value}"
            input_str += "\nLabel:"
            tuples.append((input_str.strip(), answer_choice))

        label = instance["Label"] - 1
        assert label >= 0
        assert label < len(self.answer_choices)
        return RankClassificationInstance(tuples, label)

    def instance_as_eleuther_requests(self, instance: Dict[str, Any], **kwargs):
        rci = self.instance_as_rank_classification(instance, **kwargs)
        from efficiency_benchmark.dependencies.lm_eval.base import rf

        return [rf.loglikelihood(choice[0], choice[1]) for choice in rci.choices]
