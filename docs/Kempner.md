How to run on the Kempner cluster
===

Getting started
---

1. Get your login. Log into the Kempner login nodes.
2. Set up directories:

   ```bash
   ln -s /n/holyscratch01/kempner_lab/Lab/checkpoints
   ln -s /n/holyscratch01/kempner_lab/Lab/data
   ln -s /n/holyscratch01/kempner_lab/Lab/logs
   ```

3. Add this to `~/.bashrc`. Then, run those commands in your shell.

   ```bash
   module load ncf
   module load awscli
   ```

4. `aws configure`
5. Install Miniconda with `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`, then follow prompts. You'll probably want to log out and back in after this.
6. `conda create -y -n LLM python=3.10 ipython`
7. `conda activate LLM`
8. `conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
9. `git clone https://github.com/allenai/LLM.git`
10. `cd LLM`
11. `pip install -e .`
12. Pre-download all the downstream evals. In a Python shell:

    ```bash
    from olmo.eval.downstream import *
    tokenizer = Tokenizer.from_file("tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json")
    for x in label_to_task_map.values():
        kwargs = {}
        if isinstance(x, tuple):
            x, kwargs = x
        x(tokenizer=tokenizer, **kwargs)
    ```
