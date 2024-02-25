import torch

STATEDICTS = [
    "advaveraged25.pt",
    "advaveraged2550.pt",
    "advaveraged5075.pt",
    "advaveraged75.pt",
]

sd = torch.load(STATEDICTS[0])
for state_dict in STATEDICTS[1:]:
    sd2 = torch.load(state_dict)
    for k,v in sd2.items():
        assert k not in sd
        sd[k] = v

torch.save(sd, "advaveraged.pt")
