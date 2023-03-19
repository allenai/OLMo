Experiment Log
==============

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
