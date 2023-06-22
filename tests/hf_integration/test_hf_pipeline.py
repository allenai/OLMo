from transformers import PreTrainedTokenizerFast, TextGenerationPipeline
from transformers.models.auto import AutoModelForCausalLM

from hf_integration.modeling_olmo import OLMoConfig, OLMoForCausalLM
from olmo import Tokenizer


def test_pipeline():
    # TODO: add a tiny test-fixture and use that.
    tokenizer_raw = Tokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_raw.base_tokenizer,
        truncation=tokenizer_raw.truncate_direction,
        max_length=tokenizer_raw.truncate_to,
        eos_token=tokenizer_raw.decode([tokenizer_raw.eos_token_id], skip_special_tokens=False),
    )
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    mo = OLMoForCausalLM(OLMoConfig())

    AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)
    pipeline = TextGenerationPipeline(model=mo, tokenizer=tokenizer)
    output = pipeline("question: who wrote romeo and juliet? answer: ")
    assert "generated_text" in output[0]
