
# Evaluation

We use tango and catwalk to build the pipeline.
The catwalk code exists [here](https://github.com/allenai/catwalk/tree/olmo-eval).

### Creating an evaluation config

The evaluation pipeline is run as a cross product of models that need to be evaluated, and task sets.

1. Ensure that model paths are present in a `gs://` or `s3://` location.
2. Copy `evaluation/experiments/test_config.jsonnet` to `evaluation/experiment_YYYY_MM_DD.jsonnet`
3. Add models and choose relevant task sets from [experiments/task_sets](evaluation/experiments/task_sets).

### Running the pipeline

#### Basic setup

```commandline 
export GITHUB_TOKEN="<your token>"  # Needed for beaker to clone the repo.
export GOOGLE_TOKEN="<google credentials>"  (or simply gcloud auth login) # If you are using a GS workspace.
```

#### If specifying a Google Sheet to write results to

* Share the google sheet with `olmo-eval@ai2-allennlp.iam.gserviceaccount.com`.
* Create API json key and download from [here](https://console.cloud.google.com/iam-admin/serviceaccounts/details/101308414346962828659;edit=true/keys?project=ai2-allennlp).
* Add a beaker secret:

```python
from tango.integrations.beaker.common import get_client
beaker = get_client("<beaker_workspace>")

with open("credentials_file.json") as f:
    beaker.secret.write("GDRIVE_SERVICE_ACCOUNT_JSON", f.read())
```

```commandline
export GDRIVE_SERVICE_ACCOUNT_JSON=$(cat credentials_file.json)
```

#### Run locally

```commandline
tango run evaluation/experiments/test_config.jsonnet -w your-local-workspace --include-package evaluation.steps
```

#### Run on beaker

* Update `evaluation/tango-in-beaker.yml` (the fields that should be updated are marked).

```commandline
tango --settings evaluation/tango-in-beaker.yml run evaluation/experiments/test_config.jsonnet
```

### See results

If you specify `gsheet` in your config, results will be appended to the google sheet.

All intermediate and final results will also be saved to the specified workspace, and can be accessed as follows:

```python
from tango import Workspace
workspace = Workspace.from_url("gs://your-workspace-url")
result = workspace.step_result("combine-all-outputs")
```


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

1. Add new task sets under `evaluation/experiments/task_sets` (Current full sets: `gen_tasks.libsonnet`, `eval_suite_ppl_val_v2_small.libsonnet`, `rc20_tasks.libsonnet`, `summary_tasks.libsonnet`).
2. The list of potential tasks can be seen by running `python evaluation/see_available_tasks.py`. 


#### Adding a new dataset to our perplexity eval set

1. Add the new set under our current ppl data at /net/nfs.cirrascale/allennlp/akshitab/eval_data.
2. Add the name of the folder to `experiments/task_sets/eval_suite_ppl_val_v2_small.libsonnet`

#### Adding tasks already present in catwalk

1. See `gen_tasks.libsonnet` for a simple example.

#### Adding new tasks to catwalk

(TODO: catwalk needs better documentation on adding new tasks).
1. See examples [here](https://github.com/allenai/catwalk/tree/olmo-eval/catwalk/tasks).
2. Add newly created tasks to [TASKS_LM](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/tasks_lm.py)
 or [TASKS](https://github.com/allenai/catwalk/blob/olmo-eval/catwalk/tasks/__init__.py).