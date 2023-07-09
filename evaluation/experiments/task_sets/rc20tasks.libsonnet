local task_utils = import 'task_utils.libsonnet';

local task_set = {
    name: "rc20tasks",
    tasks: ["boolq"],
    prediction_kwargs: {
        split: "validation",
        limit: 1000,
        num_shots: 0,
        num_recorded_inputs: 3,
        model_max_length: task_utils.model_max_length
    }
};


{
    task_set: task_set
}