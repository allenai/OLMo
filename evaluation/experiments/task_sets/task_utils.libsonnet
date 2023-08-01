
local create_task_set_from_task_dicts(name, task_dicts, common_kwargs) = {
    name: name,
    tasks: std.map(
        function(task_dict) common_kwargs + {
            task_name: std.get(task_dict, "task_name", std.get(common_kwargs, "task_name")),
            prediction_kwargs: std.get(common_kwargs, "prediction_kwargs", {}) + std.get(task_dict, "prediction_kwargs", {}),
            task_kwargs: std.get(common_kwargs, "task_kwargs", {}) + std.get(task_dict, "task_kwargs", {})
        },
        task_dicts
    )
};

local create_task_set_from_task_names(name, task_names, prediction_kwargs, task_kwargs) = {
    name: name,
    tasks: std.map(
                function(task_name) {
                    task_name: task_name,
                    prediction_kwargs: prediction_kwargs,
                    task_kwargs: task_kwargs
                },
                task_names
            )
};

{
    model_max_length: 2048,
    max_batch_tokens: 2048,
    create_task_set_from_task_names: create_task_set_from_task_names,
    create_task_set_from_task_dicts: create_task_set_from_task_dicts
}