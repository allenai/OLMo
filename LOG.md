Experiment Log
==============

2023-04-13
----------

Today we ran the 70B model for the first time! 32 nodes, 256 GPUs. [Some problems with the latest PyTorch](https://github.com/pytorch/pytorch/issues/97436)
prevented us from running this at full speed, and still performance was good. We only ran six batches, since this
was just about validating the setup. We will do longer runs when we have some of the obvious performance problems
sorted out.

One interesting learning is that even with just 32 nodes, nodes often don't come up cleanly. We have seen GPUs in
inconsistent states, some nodes not appearing during the rendezvous, and just undiagnosed hangs. To get a handle on
these problems, we started working on some tooling to diagnose hanging processes across the cluster, all based on
[py-spy](https://github.com/benfred/py-spy).


2023-04-03
----------

We added the option to decouple the MLP and Attention computations as in the PaLM architecture.
That is, within each transformer block we compute `MLP(LN(x)) + Attention(LN(x))` instead of `MLP(LN(x + Attention(LN(x))))` (ignoring some skip connections).
This allows to increase throughput because we can fuse the separate feed-forward and attention input projections into a single linear layer.
We also experimented with [fusing the output projections](https://github.com/allenai/LLM/pull/79) into a single linear layer but that didn't help, possibly due to the overhead of concatenating the feed-forward and attention activations together.


2023-04-02
----------

First training run! We trained a 300M model on about 70B tokens from C4.
The purpose of this model is to give the other LLM teams something in our format that's not completely random,
so they can test their evaluation and inference code.

This ran on a single node only on AMD's cluster.
On AMD hardware we're still missing Flash Attention, and we could not get `torch.compile()` to work in time for the run.
Both are expected to provide significant speedups.
This training run used model settings that are optimal for compiled models, despite not being able to compile,
because we want it to be a representative model for the downstream evaluations.


2023-03-28
----------

We've investigated a number ways to optimize training throughput in terms of tokens per second and MFU (model flop utilization). This is a list of all of the optimizations that have worked so far, ranked by how much of speedup they gave on a 1.2b param model:

1. Using FlashAttention via PyTorch's built-in `scaled_dot_product_attention` function. This resulted in a ~12% speedup over the default attention implementation while also reducing GPU memory utilization.

    Unfortunately ALiBi can't be used with FlashAttention at the moment, so the best option if we want to use relative positional encodings is probably RoPE (which can be used with FlashAttention). In general RoPE is slower than ALiBi but when combined with FlashAttention it's faster. Of course ALiBi + FlashAttention would be ideal.

1. Setting embedding/vocab size to a multiple of 128. E.g. the actual vocab size is 50257, but we force the embedding size to be 50304. This resulted in an ~11% speedup.
1. Using low-precision LayerNorm when **not** using `torch.compile()`. This resulted in a speedup of ~10%, but it actually slows throughput when using a compiled model. This probably has to do with manually casting tensors to different data types, which cause more breaks in the graph.
1. Compiling the model via `torch.compile()` with the default mode. This resulted in a ~7% speedup without increasing (and in some cases decreasing) GPU memory utilization.

    The other compile modes ("reduce-overhead" and "max-autotune") were not as fast and required substantially more GPU memory.

    Compiling as a "fullgraph" also improves throughput even further except when using FSDP since FSDP forces breaks in the graph.
1. Tweaking the FSDP settings to use "PURE" mixed precision, limit all gathers, and use non-reentrant activation checkpointing resulted in a 1-2% speedup.

Using the best compatible combination of the above settings (so everything except #3) gets us close to 60% MFU with the 1.2b model. That's really good!

For more details, see:
- [Benchmarking the performance of PyTorch's new `compile()` and built-in FlashAttention.](https://wandb.ai/ai2-llm/petew-torch2-benchmarks/reports/PyTorch-2-0-benchmarks--VmlldzozODQyMDY5?accessToken=2fh801xe265n5xx7juphb1xnx8itvls8g7nrqsjdd4ja0xlks7kaozue94z2mez3)
- [Benchmarking the cost of using RoPE](https://wandb.ai/ai2-llm/rope-benchmarks/reports/Benchmarking-RoPE--VmlldzozODQ1MjMz)
- [Benchmarking the performance of `compile()` with FSDP](https://wandb.ai/ai2-llm/fsdp-compile-benchmarks)
- [Benchmarking low precision LayerNorm](https://api.wandb.ai/links/ai2-llm/9favfpnh)


2023-03-15
----------

The cluster is down for maintenance, so we're just queueing up some features we want to run. We also used the LUMI downtime to build a better logging feature. When running 1000s of nodes in a cluster, it's difficult to get logs that make sense. We're sending our logs to third-party logging provider [logz.io](https://logz.io). It's basic, but it gets the job done.


2023-03-14
----------

This is the first day with some experiments that were serious enough to mention here.
Experiments were all on 1.3B models, with increasing microbatch size, just to shake down the system.
All runs happened on the `small-g` partition, which is for debugging only.
I reserved only one node each time.
Runtimes are limited to 15 minutes, which is too short for performance to stability, but good enough to get an idea.
Findings:
 * WandB integration works.
 * Launching nodes with slurm works.
   We let slurm launch everything, but there is a different school of thought that says that slurm should just launch one process on each node, and then you use `torch.distributed` to spread out on the node.
   I'm not sure what that buys us, and it's one extra component in the mix, so I didn't do it that way.
 * Automatic restarts work. One run got killed and automatically restarted.
   It is great that restarts work, but somewhat worrisome that we're already sampling this behavior after less than 45 minutes of runtime on only one node.
