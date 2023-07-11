
# Evaluation

We use tango and catwalk to build the pipeline.
The catwalk code exists [here](https://github.com/allenai/catwalk/tree/olmo-eval).

### Creating an evaluation config

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
3. See `eval_suite` for an example where each task needs to be modified with `task_kwargs`.

### Running the pipeline

#### Run on beaker


`tango --settings evaluation/tango-in-beaker.yml run evaluation/experiments/test_config.jsonnet`
