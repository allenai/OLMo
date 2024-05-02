Experiment Log
==============

2023-07-12
----------

For about a week, we have been chasing an issue where our loss curve looks wavy like this:
<img width="519" alt="Screenshot 2023-07-13 at 14 56 19" src="https://github.com/allenai/LLM/assets/920638/5fec3ad9-5fd6-4959-956d-9f47e5232bd2">

Our colleagues from MosaicML suggested that our data might not be properly mixed, but we reviewed the code carefully and
found no problems. However, after exhausting all other possibilities, we had nothing left to go on, so we decided
to try and graph our batch composition over time. Turns out, there are significant changes in batch composition after all:

![image](https://github.com/allenai/LLM/assets/920638/3362e78e-4554-451e-8a59-a0114a4c4d56)

In this graph, organge is content from Common Crawl, and green is content from The Stack, i.e., code. As you can see, the
proportion of code changes significantly over time, and if you overlay the graphs, you can see that more code means lower
loss. So clearly something is up with our shuffling after all.

When we construct batches, we concatenate all content into one giant array of instances (samples), and then shuffle the
array. We use `torch.randperm()` to shuffle. Long story short, it turns out that `torch.randperm()` does not shuffle very
well. When you graph the index of the instances that end up in our batches over time, you see a very pronounced pattern:

![image](https://github.com/allenai/LLM/assets/920638/39b01f8d-f1db-4485-b339-c20ee423b98a)

While it would be interesting to find out why this happens, we left that as an exercise for the PyTorch team, and
re-implemented our shuffling code to use NumPy. Now the curve looks like this:

![image](https://github.com/allenai/LLM/assets/920638/192c5790-ab1f-4a3d-8fb6-a9dbc74391e8)

Nice and random!

![image](https://imgs.xkcd.com/comics/random_number.png)



2023-04-26
----------

Large node counts means large failure rates. Yesterday's run ran out of memory at 3am, so we didn't get as many
batches done as we were hoping. Today's issues:
 * Something seems to be wrong with the super-fast network drive we're using for checkpointing. Writing checkpoints
   there consistently fails today, even though it worked fine yesterday. We switched to a slower drive for
   checkpointing for now to make some progress.
 * Occasionally the Slingshot interconnect between the nodes fails with the message "Cassini Event Queue overflow".
   The workaround is to set a larger event queue size by setting the environment variable
   `FI_CXI_DEFAULT_CQ_SIZE` to `131072` (or some other large number). 
 * A lot of nodes fail to come up when starting a job. There are at least two separate issues that cause this.
   Low-level failures in RCCL or device drivers do not get logged properly. Instead, they just print to stdout and
   stderr. We altered our launch script to nevertheless tell us which node is reporting which error
   (a078ae4686e190dc1e9eb91ab8f434e90d95d152). Using this, we can exclude nodes from the launch every time one
   errors out during startup. It's a laborious process. Starting a 64-node job will often take 30 minutes or more
   due to these failures. To get a better handle on the situation we started a spreadsheet that keeps track of the
   failures. If we start to see a pattern, maybe we can do something more intelligent than excluding nodes one at
   a time.

Despite all this, the model is now training properly, and at the time of writing we have trained on 7B tokens.
We even had our first proper loss spike!

<img width="557" alt="Screenshot 2023-04-26 at 16 51 59" src="https://user-images.githubusercontent.com/920638/234726481-2fceb391-65da-4da9-9844-aaf0c493ee6a.png">


2023-04-25
----------

The issues with checkpointing have been resolved, and we have a big training run under way. We're using this
opportunity to track speed vs. number of nodes.

<img width="479" alt="Screenshot 2023-04-26 at 16 55 55" src="https://user-images.githubusercontent.com/920638/234726824-074e6386-7e8a-4ec2-9afd-38717d2e601d.png">

The results are pretty good. We lose about 20% efficiency to communication overhead, which is acceptable.
With 64 nodes we no longer have to do gradient accumulation, so it's possible that's why the 64-node configuration
is actually faster than the 32-node one.

2023-04-24
----------

We're finding more and more problems with Torch 2. We have to use Torch 2 because some drivers that make our
hardware work are only available for Torch 2, but it seems really half-baked in its current state. Compounding the
problems is the fact that we're attempting to use MosaicML's Composer to run the training, but Torch 2 is not
officially supported by Composer yet. In an effort to not stack two unstable bits of software on top of each other,
we decided to write our own trainer instead of relying on Composer.

While we're tracking a number of smaller problems around `torch.compile()` and random numbers, the big issue is
checkpointing. Writing a model that's using Torch's FSDP to disk is surprisingly difficult.


2023-04-18
----------

While not strictly necessary from a scientific perspective, we thought it might be a good idea to train the
medium size model to 300B tokens, to shake down the system, and make sure we're on the right track. There was no
way we could have the data done in time for this run, so we're just training on C4. However, in an effort to make
this run useful as a baseline, we wanted to have a somewhat reasonable validation set. We chose the recently
released Red Pajama data, because it is quite close to what we want to do with our own data. The data team is
working on this right now.


2023-04-17
----------

Our original model settings and sizes came from MosaicML. We found they are a lot smaller than they say they are,
so we went back to the [PaLM paper](https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html)
and put in those settings. Most of our architecture choices follow the PaLM paper, so it makes sense to do it with
the dimensions of the model as well.

Since the new sizes don't follow exactly the 1B/7B/70B scheme, we now call them "small", "medium", and "large".
The new sizes are as follows:

| Name             | parameters -h |  parameters | non-embedding parameters |
|------------------|--------------:|------------:|-------------------------:|
| extra-tiny-debug |          16 M |    16169216 |                  3291392 |
| tiny             |         288 M |   288706560 |                237195264 |
| small            |            1B |  1051439104 |                948416512 |
| medium           |         7.8 B |  7791353856 |               7585308672 |
| large            |        60.8 B | 60818014208 |              60405923840 |


2023-04-13
----------

We've been experimenting with a [triton](https://github.com/openai/triton) implementation of [FlashAttention](https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py) that supports using an arbitrary attention bias, which would allow us to use [ALiBi](https://www.semanticscholar.org/paper/f5aba74fbd512190ed5f61127618381f70710572).
Unfortunately it doesn't look like this is going to be a viable option at the moment.
This particular implementation only works on an older version of triton that uses a CUDA-specific backend.
Therefore it won't run on AMD GPUs.

We'll revisit this again when there are updates to [HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention).

Meanwhile, we ran the 70B model for the first time! 32 nodes, 256 GPUs. [Some problems with the latest PyTorch](https://github.com/pytorch/pytorch/issues/97436)
prevented us from running this at full speed, and still performance was good. We only ran six batches, since this
was just about validating the setup. We will do longer runs when we have some of the obvious performance problems
sorted out.

One interesting learning is that even with just 32 nodes, nodes often don't come up cleanly. We have seen GPUs in
inconsistent states, some nodes not appearing during the rendezvous, and just undiagnosed hangs. To get a handle on
these problems, we started working on some tooling to diagnose hanging processes across the cluster, all based on
[py-spy](https://github.com/benfred/py-spy).


2023-04-12
----------

Today we got the [Slingshot Interconnect](https://www.hpe.com/us/en/compute/hpc/slingshot-interconnect.html)
working. The LUMI cluster uses this style of interconnect to tie the GPUs together with an aggregate of 800GBit/s.
For a large distributed training job, the speed of this interconnect is absolutely essential. Case in point, the
1.2B model, on only two nodes (16 GPUs), went from 7500 tokens/second/GPU to 9500 tokens/second/GPU. That is a huge
increase in speed!

In the details, getting this to work was all about using the right libraries and making sure they are available to
the right process at the right time. This is all about setting the right environment variables, setting the right
flags, and general low-level Linux stuff. It's not the sexy part of training large language models.


2023-04-11
----------

PyTorch 2.0 came with a new feature: `torch.compile()`. It promises massive speedups if you set up your model
right. We intend to take advantage, and got it working with NVidia hardware, but it was a bit harder to make it
work on AMD hardware as well. With the help of AMD engineers we figured it out today, and immediately saw a 15%
speedup on the 1.2B model! 15% are hard to come by, so this is a big success.


2023-04-10
----------

Today we got the 7B model running on LUMI for the first time. This ran on 8 nodes, 64 GPUs. We're missing a lot of
tricks to make it fast, and yet we saw 1200 tokens/second/GPU. That's pretty good!

It took a long time to get this working, mainly due to an issue that had nothing to do with the 7B config. ALiBi
attention uses some constant static tensors for its calculations. We don't want to recompute these for every batch,
so we keep them in torch "buffers", which are like parameters, except they don't receive gradient updates.
These buffers are in bf16 format, and contain a lot of `-inf`, so right off the bat they are exploring a lot of
edge cases. Also, [torch buffers are poorly defined in the context of FSDP](https://github.com/pytorch/pytorch/blob/4d3d3317ebd1c57a28754281d91ed7f35f4ce320/torch/distributed/fsdp/_init_utils.py#L257),
and some of the operations that FSDP does on them result in `NaN`. The solution is to [store these tensors in a
different way so that FSDP does not interfere with them](https://github.com/allenai/LLM/pull/90/files#diff-ef8ab7279deeec716e70a1cc9ab2accaaa60f27b301cc0733f1e00a9e39c07d1).


2023-04-06
----------

The LUMI cluster is back from maintenance! This is one day earlier than expected. LUMI software has been updated
as part of the maintenance, and it broke our launch scripts. On these compute nodes, the topology is such that
every GPU has a fast connection to 8 CPUs, and a somewhat slower connection to the others. So we use "CPU binding"
to make sure the right CPUs are paired up with the right GPUs. This part of the system broke, so for now we're
running without it. We never benchmarked its impact, so it's not clear how important it really is. The bandwidth
from CPU to GPU isn't the biggest bottleneck anyways, so it's probably not a problem.


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
