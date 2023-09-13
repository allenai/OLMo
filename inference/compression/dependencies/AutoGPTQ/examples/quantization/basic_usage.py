import os

from transformers import AutoTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


pretrained_model_dir = "facebook/opt-125m"
quantized_model_dir = "opt-125m-4bit-128g"


def main():
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
    examples = [
        tokenizer(
            "auto-gptq is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
        )
    ]

    quantize_config = BaseQuantizeConfig(
        bits=4,  # quantize model to 4-bit
        group_size=128,  # it is recommended to set the value to 128
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
    )

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)

    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(examples)

    # save quantized model
    model.save_quantized(quantized_model_dir)

    # push quantized model to Hugging Face Hub.
    # to use use_auth_token=True, Login first via huggingface-cli login.
    # or pass explcit token with: use_auth_token="hf_xxxxxxx"
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)

    # alternatively you can save and push at the same time
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    # commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, save_dir=quantized_model_dir, use_safetensors=True, commit_message=commit_message, use_auth_token=True)

    # save quantized model using safetensors
    model.save_quantized(quantized_model_dir, use_safetensors=True)

    # load quantized model to the first GPU
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0")

    # download quantized model from Hugging Face Hub and load to the first GPU
    # model = AutoGPTQForCausalLM.from_quantized(repo_id, device="cuda:0", use_safetensors=True, use_triton=False)

    # inference with model.generate
    print(tokenizer.decode(model.generate(**tokenizer("auto_gptq is", return_tensors="pt").to(model.device))[0]))

    # or you can also use pipeline
    pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)
    print(pipeline("auto-gptq is")[0]["generated_text"])


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    main()
