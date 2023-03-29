Experiment Log
==============

2023-03-28
----------

We've investigated a number ways to optimize training throughput in terms of tokens per second and MFU (model flop utilization). This is a list of all of the optimizations that have worked so far, ranked by how much of speedup they gave on a 1.2b param model:

1. Using FlashAttention via PyTorch's built-in `scaled_dot_product_attention` function. This resulted in a ~12% speedup over the default attention implementation while also reducing GPU memory utilization.

    Unfortunately ALiBi can't be used with FlashAttention at the moment, so the best option if we want to use relative positional encodings is probably RoPE (which can be used with FlashAttention). In general RoPE is slower than ALiBi but when combined with FlashAttention it's faster. Of course ALiBi + FlashAttention would be ideal.

1. Setting embedding/vocab size to a multiple of 128. E.g. the actual vocab size is 50257, but we force the embedding size to be 50304. This resulted in an ~11% speedup.
1. Using low-precision LayerNorm when **not** using `torch.compile()`. This resulted in a speedup of ~10%, but it actually slows throughput when using a compiled model.
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

2023-03-15
----------

The cluster is down for maintenance, so we're just queueing up some features we want to run. We also used the LUMI downtime to build a better logging feature. When running 1000s of nodes in a cluster, it's difficult to get logs that make sense. We're sending our logs to third-party logging provider [logz.io](https://logz.io). It's basic, but it gets the job done.
