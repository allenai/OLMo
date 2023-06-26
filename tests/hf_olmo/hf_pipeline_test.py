def test_pipeline(model_path: str):
    from transformers import TextGenerationPipeline
    from transformers.models.auto import AutoModelForCausalLM, AutoTokenizer

    from hf_olmo.modeling_olmo import OLMoConfig, OLMoForCausalLM  # noqa: F401

    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    output = pipeline("question: who wrote romeo and juliet? answer: ", max_new_tokens=30)
    assert "generated_text" in output[0]
