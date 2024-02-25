import os

import torch

CKPTS = [
    "step456000-unsharded-lumi/model.pt",
    "step556000-unsharded-mosaic/model.pt"
]

OUTDIR = "step456000-unsharded-lumi-mosaic"

CKPTS = [
    "step456000-unsharded-lumi/model.pt",
    "step432410-unsharded-mosaic/model.pt"
]






import os

import torch

CKPTS = [
"step551000-unsharded/model.pt",
"step552000-unsharded/model.pt",
"step553000-unsharded/model.pt",
"step554000-unsharded/model.pt",
"step555000-unsharded/model.pt",
"step556000-unsharded/model.pt",
"step557000-unsharded/model.pt",
]

OUTDIR = "last7_avg"

first_sd = torch.load(CKPTS[0])
for k in first_sd:
    first_sd[k] = torch.stack([sd[k] for sd in [torch.load(ckpt) for ckpt in CKPTS]], dim=0).mean(dim=0)

os.makedirs(OUTDIR, exist_ok=True)
torch.save(first_sd, os.path.join(OUTDIR, "model.pt"))





import os

import torch

CKPTS = [
"step551000-unsharded/model.pt",
"step552000-unsharded/model.pt",
"step553000-unsharded/model.pt",
"step554000-unsharded/model.pt",
"step555000-unsharded/model.pt",
"step556000-unsharded/model.pt",
"step557000-unsharded/model.pt",
]

OUTDIR = "last7_avg"

keys = list(torch.load(CKPTS[0]).keys())
new_sd = {}
for k in keys:
    new_sd[k] = torch.stack([torch.load(ckpt)[k] for ckpt in CKPTS], dim=0).mean(dim=0)

os.makedirs(OUTDIR, exist_ok=True)
torch.save(new_sd, os.path.join(OUTDIR, "model.pt"))
