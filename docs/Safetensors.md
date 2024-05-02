That weird safetensors format
===

Short explanation
---

When you want to load state dicts from disk:

1. Convert your `model.pt` and `optim.pt` files to `.safetensors` files with [safetensors_util.py](../olmo/safetensors_util.py).
   Needs a lot of memory.
2. Copy the resulting files next to the `.pt` file in the checkpoint directory. When you load from this checkpoint,
   OLMo will see it there and load the files really fast and with very little CPU memory overhead.


Long explanation
---

OLMo saves unsharded checkpoints with `torch.save()`, which writes a state dictionary to a file using (essentially)
`pickle` from the Python standard library. The problem with this is that `pickle` is slow, single threaded, and
forces us to read in the entire file before reading any part of it. For the 65B model, checkpoints are about 700GB
and take many minutes to load, so this is a significant problem. Furthermore, when we're running on a machine with
8 GPUs, we can't load the model 8 times in parallel, because we will run out of memory.

[Huggingface's safetensors format](https://github.com/huggingface/safetensors) solves these problems. Safetensors
gives us a way to store state dictionaries in such a way that when you load them, the tensors' data will be
memory mapped from the disk. That means that the tensor won't actually be loaded until your code actually accesses
it. Multiple processes loading the same safetensors file on the same machine (which is exactly what happens when OLMo loads
a model from a checkpoint), will read the data only once.

Unfortunately, safetensors does not go far enough. Safetensors can do its magic if you have a state dict that
conforms to the type `Dict[str, Tensor]`, i.e., a Python dictionary mapping strings to tensors. This holds for
model weights, but it does not hold for optimizer state. So we put a layer on top of safetensors that maps the
data types that OLMo needs for model and optimizer state to the data types that safetensors needs to do its magic.
This mapping happens in [safetensors_util.py](../olmo/safetensors_util.py).

The key functions are:
 * `state_dict_to_safetensors_file(state_dict: Dict, filename: PathOrStr)`, which writes a Python
   dictionary to a file using this format. This takes a normal amount of time. There is no way around the fact
   that you have to write all that data to disk, even with safetensors.
 * `safetensors_file_to_state_dict(filename: PathOrStr)`, which reads a Python dictionary from a file in this
   format. This is abnormally fast, a few seconds at most, even with a 250GB file, because it does not really read
   the data until it is accessed.

There is a script that can take a file in PyTorch's normal format and convert it into this special safetensors
format: [convert_pt_to_safetensors.py](../scripts/convert_pt_to_safetensors.py). This script has to load the
original file the slow way, and store the content in memory in its entirety, so it takes a lot of memory and time.

When it's time to load the file, whenever OLMo tries to load a state dictionary from file of the name `foo.pt`, it
first checks whether the file `foo.safetensors` exists. If so, it loads that one instead.