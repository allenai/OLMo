import torch
from hf_olmo import *
from vllm import SamplingParams, LLM
from vllm.model_executor.utils import set_random_seed
from transformers import AutoModelForCausalLM, AutoTokenizer

def compare_vllm_with_hf(path: str, prompt: str = "My name is John! I am ", which: str = "both"):

    assert which in ["both", "hf", "vllm"]
    
    # VLLM
    if which in ["vllm", "both"]:
        s = SamplingParams(temperature=0.0)
        llm = LLM(model=path, trust_remote_code=True, gpu_memory_utilization=0.90)

        set_random_seed(0)
        print(llm.generate([prompt], sampling_params=s))

    # HF
    if which in ["hf", "both"]:
        hf_model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, device_map="auto").cuda()
        tokenizer = AutoTokenizer.from_pretrained(path)
        input = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input).unsqueeze(0)

        set_random_seed(0)
        hf_gen = hf_model.generate(input_tensor.long().cuda())

        print(tokenizer.decode(hf_gen[0].tolist()))


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 2:
        compare_vllm_with_hf(sys.argv[1])
    else:
        compare_vllm_with_hf(sys.argv[1], which=sys.argv[2])
