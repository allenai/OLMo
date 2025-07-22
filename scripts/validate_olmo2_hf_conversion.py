import argparse
import math
from transformers import Olmo2ForCausalLM, AutoTokenizer
import torch

def load_model_and_tokenizer(path, device, revision=None):
    model = Olmo2ForCausalLM.from_pretrained(path, revision=revision).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
    return model, tokenizer

def generate_output(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=25,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_tokens_b(step, batch_size=512, seq_len=4096):
    total_tokens = step * batch_size * seq_len
    tokens_b = math.ceil(total_tokens / 1_000_000_000)
    return tokens_b

def main(step, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load local model
    local_path = f"/weka/baileyk/OLMo-medium/peteish1-early-checkpoints-hf/step{step}-unsharded-hf"
    print(f"Loading local model from {local_path} on {device}")
    local_model, local_tokenizer = load_model_and_tokenizer(local_path, device)
    local_output = generate_output(local_model, local_tokenizer, prompt, device)

    # Compute token count and load HF model
    tokens_b = compute_tokens_b(step)
    hf_path = "allenai/OLMo-2-0425-1B"
    hf_revision = f"stage1-step{step}-tokens{tokens_b}B"
    print(f"Loading HF model from {hf_path} (revision: {hf_revision})")
    hf_model, hf_tokenizer = load_model_and_tokenizer(hf_path, device, revision=hf_revision)
    hf_output = generate_output(hf_model, hf_tokenizer, prompt, device)

    # Display comparison
    print("\nPrompt:")
    print(prompt)
    print("\nLocal Output:")
    print(local_output)
    print("\nHF Output:")
    print(hf_output)

    if local_output.strip() == hf_output.strip():
        print("\nMatch: Outputs are identical.")
    else:
        print("\nMismatch: Outputs differ.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step number (e.g. 1000)")
    parser.add_argument("--prompt", type=str, default="What is the capital of the United States?", help="Prompt text to use")
    args = parser.parse_args()
    main(args.step, args.prompt)

