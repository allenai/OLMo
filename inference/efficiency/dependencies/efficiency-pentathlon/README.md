# Efficiency Pentathlon
Efficiency Pentathlon is a standardized benchmark to evaluate the inference efficiency of NLP models.
It offers a controlled hardware platform, and is designed to mirror real-world applications scenarios. 
It provides a suite of metrics that target different aspects of efficiency, including latency, throughput, memory overhead, and energy consumption.

## Installing

### Installing from source

1. **Set up a Conda environment for `efficiency-pentathlon`**

```bash
conda create -n efficiency-pentathlon python=3.8
conda activate efficiency-pentathlon
```

2. **Clone this repository**
```bash
git clone https://github.com/allenai/efficiency-pentathlon.git
cd efficiency-pentathlon/
```

3. **Install pentathlon**

```bash
pip install .
```

4. **Avoid dependency conflicts**
`efficiency-pentathlon` installs `PyTorch` as a dependency, which might conflict with your project's `PyTorch`. A workaround is to use `efficiency-pentathlon` with its absolute path:

```bash
export EP=$(which efficiency-pentathlon)
```
You can now use `$EP` in a different Conda virtual environment for your repository.

## Local evaluation.

1. **Set up your codebase**
Create a conda enviroment for your codebase and install the dependencies. As a running example, let's consider [this machine translation repository](https://github.com/haopeng-nlp/submission). 
Let's create a new conda environment to avoid any potential conflicts among the dependencies of `efficiency-pentathlon` and this repository. 

```bash
conda create -n mt-efficiency python=3.8
conda activate mt-efficiency
git clone https://github.com/haopeng-nlp/submission.git
cd submission/
pip install -r requirements.txt
```

2. **Use `efficiency-pentathlon run` to evaluate inference efficiency locally**
We are now under the `mt-efficiency` conda environment, but can still access `efficiency-pentathlon` by `$EP` as a result of step 4 in the above section.

```bash
$EP run --task wmt16-ro-en --scenario single_stream -- python entrypoint.py --model facebook/mbart-large-50-many-to-many-mmt --task wmt16-ro-en
```

The above command line evaluates the specified model on `wmt16-ro-en` translation task with the `single_stream` scenario.

Pentathlon currently supports four evaluation scenarios `single_stream`, `fixed_batch`, `random_batch`, and `offline`. 
You can specify the scenario with the `--scenario` argument. More on this later.

Built upon [Catwalk](https://github.com/allenai/catwalk), pentathlon supports all tasks that catwalk does. Taking one step further, pentathlon also supports any dataset that can be loaded through the [Huggingface's `datasets.load_dataset` API](https://huggingface.co/docs/datasets/v1.11.0/loading_datasets.html). This can be achieved by the `--hf_dataset_args` argument. For example, for the same task as the above:

```bash
$EP run --hf_dataset_args '{"path": "wmt16", "name": "ro-en"}' --scenario single_stream -- python entrypoint.py --model facebook/mbart-large-50-many-to-many-mmt --task wmt16-ro-en
```

*❗Everything after the `--` is the command + arguments you want to run your repository code with. It's important to include the `--` so pentathlon can differentiate them from its own options.*

## To submit to and evaluate the model on a dedicated machine
Pentathlon includes a dedicated in-house server hosted by AI2, allowing participants to evaluate their models in a controlled hardware environment. The submission process to this server is similar to that of local evaluation but uses `submit` instead of `run`.

1. **Prepare a GitHub repository for your project**
If you haven't already, create a GitHub repository for your project, clone it locally, and then navigate to the root directory of this repository.
**The submission process must be done from the root directory of your repository.**

2. **Submission**
This process is very similar to the local evaluation we've just seen, but replaces `run` with `submit`:

```bash
$EP submit --task wmt16-ro-en --scenario single_stream -- python entrypoint.py --model facebook/mbart-large-50-many-to-many-mmt --task wmt16-ro-en
```

*❗At this moment, the evaluation on the the AI2 server is only available for AI2 practitioners. We are finalizing the paperworks, and will make the it public to all very soon.*

## Preparin Efficiency Pentathlon for efficiency evaluation

### Standard input and ouput interaction
Pentathlon interacts with the model to be evaluated through `stdio`--input instances are written to the stdin of the model process, and the pentathlon reads from the model process's `stdout` its outputs. 
No assumption is made on the model's, e.g., deep learning framework (if a deep learning model is used at all) or the programming language it's implemented in. 
Here is a [Python example](https://github.com/haopeng-nlp/submission/blob/main/entrypoint.py#L20-L76) to handle the 
stdio interactions. 
In this case, Pentathlon requires that two methods are implemented:

1.  [`predict`](https://github.com/haopeng-nlp/submission/blob/main/entrypoint.py#L33). It takes as input a list of instances (each with type `Dict[str, Any]`), and returns the output as a list of strings. 
2.  [`prepare`](https://github.com/haopeng-nlp/submission/blob/main/entrypoint.py#L56). **Specifically for the `offline` scenario**, it prepares the model for evaluation. such as loading the checkpoint and tokenizer. After this method is executed, the model should be ready to receive inputs and produce outputs, and pentathlon starts the timer.
   
### Preparing the checkpoint for submission to our machine
To submit your model to our machine for controled evaluation, code to download the model's checkpoint needs to be implemented.
This can be done through, e.g., [Hugging Face model hub](https://huggingface.co/docs/hub/models-the-hub), [`git-lfs`](https://git-lfs.com/), or the cloud service.