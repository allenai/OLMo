
# Evaluation

We use tango and catwalk to build the pipeline.
The catwalk code exists [here](https://github.com/allenai/catwalk/tree/olmo-eval).

### Creating an evaluation config

The evaluation pipeline is run as a cross product of models that need to be evaluated, and task sets.

1. Ensure that model paths are present in a `gs://` or `s3://` location.
2. Copy `evaluation/test_config.jsonnet` to `evaluation/experiment_YYYY_MM_DD.jsonnet`
3. Add models and choose relevant task sets from [evaluation/experiments/task_sets](evaluation/experiments/task_sets).

### Adding new task sets

A task set is of the form:

```jsonnet
{
    name: "<Name of the task set>",
    tasks: [
        {
            task_name: "<One of the tasks present in `TASKS_LM` or `TASKS`>",
            task_kwargs: "<task-specific kwargs (See eval_suite for examples)>",
            prediction_kwargs: "<kwargs on how to evaluate the model on this task>"
        }
    ]
}
```

1. Add new task sets under `evaluation/experiments/task_sets` (Examples: `gentasks.libsonnet`, `eval_suite_ppl_val_v2_small.libsonnet`). 
2. See `gentasks.libsonnet` for a simple example (should cover most cases).
3. See `eval_suite_ppl_val_v2_small` for an example where we use our custom perplexity eval set.
4. The list of potential tasks are in [TASKS_LM](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/tasks_lm.py)
 and [TASKS](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/__init__.py). New tasks should be added here.

### Running the pipeline

### Setup

```commandline
gcloud auth login  # This will automatically set `GOOGLE_TOKEN` to your default credentials. 
export GITHUB_TOKEN="<your token>"  # Needed for beaker to clone the repo.
```

If specifying a Google Sheets to write to:

* Share the google sheet with `olmo-eval@ai2-allennlp.iam.gserviceaccount.com`.
* Create API json key and download from [here](https://console.cloud.google.com/iam-admin/serviceaccounts/details/101308414346962828659;edit=true/keys?project=ai2-allennlp).
* Add a beaker secret:

```python
from beaker import Beaker
b = Beaker.from_env()

with open("credentials_file.json") as f:
    b.secret.write("GDRIVE_SERVICE_ACCOUNT_JSON", f.read())
```

#### Run on beaker

1. Update `evaluation/tango-in-beaker.yml`.
2. Run your eval configuration.

```commandline
tango --settings evaluation/tango-in-beaker.yml run evaluation/experiments/test_config.jsonnet
```

### See results

If you specify `gsheet` in your config, results will be appended to the google sheet.

All intermediate and final results will also be saved to the specified workspace, and can be accessed as follows:

```python
from tango import Workspace
workspace = Workspace.from_url("gs://your-workspace-url")
result = workspace.step_result("outputs_as_rows_<model_name>_<task_set>")
```
