
local task_utils = import 'task_utils.libsonnet';

local task_set = {
    name: "gentasks",
    tasks: ["drop"],
    prediction_kwargs: {
        split: "validation",
        limit: 1000,
        num_shots: 5,
        fewshot_seed: 1234,
        num_recorded_inputs: 3,
        model_max_length: task_utils.model_max_length,
    }
};

{
    task_set: task_set
}