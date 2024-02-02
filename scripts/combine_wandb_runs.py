from pathlib import Path

import wandb
from wandb.wandb_run import Run

runs_path = Path("~/Downloads/OLMo Runs - mitchish-mcli.csv").expanduser()
dst_entity = "ai2-llm"
dst_project = "OLMo-7B"
dst_group = "OLMo-7B"
dst_run_name = "OLMo-7B"

# Read CSV dump from "OLMo Runs" spreadsheet.
#  run_paths_to_copy = []
#  with runs_path.open("r") as f:
#      reader = csv.DictReader(f, delimiter=",")
#      for row in reader:
#          run_paths_to_copy.append(row["W&B URL"].replace("https://wandb.ai/", ""))

#      # Order oldest -> newest.
#      run_paths_to_copy = list(reversed(run_paths_to_copy))

run_paths_to_copy = [
    "ai2-llm/olmo-medium/runs/wvc30anm",
    "ai2-llm/olmo-medium/runs/uhy9bs35",
    "ai2-llm/olmo-medium/runs/l6v218f4",
    "ai2-llm/olmo-medium/runs/8fioq3qx",
    "ai2-llm/olmo-medium/runs/mk9kaqh0",
    "ai2-llm/olmo-medium/runs/49i87wpn",
    "ai2-llm/olmo-medium/runs/0j2eqydw",
    "ai2-llm/olmo-medium/runs/5wkmhkqh",
    "ai2-llm/olmo-medium/runs/hrshlkzq",
    "ai2-llm/olmo-medium/runs/eysi0t0y",
    "ai2-llm/olmo-medium/runs/7gomworq",
    "ai2-llm/olmo-medium/runs/lyij2l8m",
    "ai2-llm/olmo-medium/runs/99euueq4",
    "ai2-llm/olmo-medium/runs/fcn5q3zw",
    "ai2-llm/olmo-medium/runs/j18wauyq",
    "ai2-llm/olmo-medium/runs/jtfwv96r",
    "ai2-llm/olmo-medium/runs/yuc5kl7s",
    "ai2-llm/olmo-medium/runs/25urleov",
    "ai2-llm/olmo-medium/runs/obde4w9j",
    "ai2-llm/olmo-medium/runs/eaqax5ns",
    "ai2-llm/olmo-medium/runs/cojbrc1o",
    "ai2-llm/olmo-medium/runs/4xel5n7e",
    "ai2-llm/olmo-medium/runs/jcs4c32w",
    "ai2-llm/olmo-medium/runs/x55jyv7k",
    "ai2-llm/olmo-medium/runs/yv7lgx0i",
    "ai2-llm/olmo-medium/runs/11uf7gsv",
    "ai2-llm/olmo-medium/runs/lds6zcog",
    "ai2-llm/olmo-medium/runs/ho7jy4ey",
    "ai2-llm/olmo-medium/runs/87shig0a",
    "ai2-llm/olmo-medium/runs/x6zdcp5j",
    "ai2-llm/olmo-medium/runs/olocmvn0",
    "ai2-llm/olmo-medium/runs/xtruaap8",
    "ai2-llm/olmo-medium/runs/2l070ogq",
    "ai2-llm/olmo-medium/runs/uy2ydw12",
    "ai2-llm/olmo-medium/runs/x23ciyv9",
    "ai2-llm/olmo-medium/runs/67i5mdg0",
    "ai2-llm/olmo-medium/runs/wrv46m83",
    "ai2-llm/olmo-medium/runs/wd2gxrza",
    "ai2-llm/olmo-medium/runs/z4z0x4m9",
    "ai2-llm/olmo-medium/runs/p067ktg9",
]

# Set your API key
wandb.login()

# Initialize the wandb API
api = wandb.Api()

# Iterate through the runs and copy them to the destination project
for i, run_path in enumerate(run_paths_to_copy):
    run = api.run(run_path)

    print(f"Copying run {i+1} of {len(run_paths_to_copy)}")

    # Get the run history and files
    history = run.history()

    # Create a new run in the destination project
    new_run = wandb.init(
        project=dst_project,
        entity=dst_entity,
        config=run.config,
        name=f"{dst_run_name}-run-{i+1:03d}",
        resume="allow",
        group=dst_group,
    )
    assert isinstance(new_run, Run)

    # Log the history to the new run
    for index, row in history.iterrows():
        new_run.log(row.to_dict())

    # Finish the new run
    new_run.finish()
