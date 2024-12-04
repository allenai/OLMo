import tempfile
from typing import Iterable

import pandas as pd
import wandb
from wandb.apis.public import Run as RunApi
from wandb.errors import CommError
from wandb.sdk.wandb_run import Run

ENTITY = "ai2-llm"
# (run_path, new_run_name, new_run_group, new_project, step_offset)
# step_offset: This is added to the run's step numbers. It is useful for placing anneals
#     at the end of their corresponding runs.
RUNS_TO_COPY = [
    ###########################################################################################
    (
        "ai2-llm/olmo-medium/runs/79h5h3aa",
        "peteish7-weka-anneal-from-928646-50B-nowup_legal-whammy-2_seed42",
        "peteish7-weka-anneal-from-928646-50B-nowup_legal-whammy-2_seed42",
        "olmo-7b-anneals",
        0
    ),

]


def get_history(run: RunApi, api: wandb.Api) -> Iterable[dict]:
    try:
        # Try to use the W&B artifact containing the full run history
        history_artifact = api.artifact(f"{run.entity}/{run.project}/run-{run.id}-history:latest")

        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = str(history_artifact.file(str(temp_dir)))

            if not history_path.endswith(".parquet"):
                raise RuntimeError(
                    f"Expected history of run {run.id} to have parquet extension, got {history_path}"
                )

            history = pd.read_parquet(history_path, engine="pyarrow")
            return [step_data.dropna().to_dict() for _, step_data in history.iterrows()]
    except CommError:
        # Sometimes W&B runs don't seem to have an artifact with the run history. In that case, fall
        # back to calling the api for the run history.
        return run.scan_history()


def main():
    # Set your API key
    wandb.login()

    # Initialize the wandb API
    api = wandb.Api()

    # Iterate through the runs and copy them to the destination project
    for run_path, new_run_name, new_run_group, new_project, step_offset in sorted(RUNS_TO_COPY, key=lambda x: x[1]):
        run = api.run(run_path)
        assert isinstance(run, RunApi)

        print(f"Copying run '{run_path}' to '{new_project}/{new_run_name}'...")

        # Get the run history and files
        history = get_history(run, api)

        # Create a new run in the destination project
        new_run = wandb.init(
            project=new_project,
            entity=ENTITY,
            config=run.config,
            name=new_run_name,
            resume="allow",
            group=new_run_group,
            settings=wandb.Settings(_disable_stats=True),
        )
        assert isinstance(new_run, Run)

        # Log the history to the new run
        for data in history:
            step = int(data.pop("_step"))

            # Data typically seems to be empty for step 0 (just GPU usage, for example)
            if step == 0:
                continue

            step += step_offset

            new_run.log(data, step=step)

        # Finish the new run
        new_run.finish()


if __name__ == "__main__":
    main()
