
local task_utils = import '../task_utils.libsonnet';

local name = "rc20_tasks";
local task_names = ["boolq", "arc_easy"];
local prediction_kwargs = {
    split: "validation",
    limit: 1000,
    num_shots: 0,
    num_recorded_inputs: 3,
    //model_max_length: task_utils.model_max_length
    model_max_length: 256
};
local task_kwargs = {};

{
    task_set: task_utils.create_task_set_from_task_names(name, task_names, prediction_kwargs, task_kwargs)
}
