def test_pipeline(model_path: str):
    from transformers import TextGenerationPipeline

    from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast

    model = OLMoForCausalLM.from_pretrained(model_path)
    tokenizer = OLMoTokenizerFast.from_pretrained(model_path)
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    output = pipeline("question: who wrote romeo and juliet? answer: ", max_new_tokens=30)
    assert "generated_text" in output[0]
