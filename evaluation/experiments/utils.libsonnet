

/*

task_set: {
    name: task_set_name,
    task_dicts: [
        {task: task_name, prediction_kwargs: prediction_kwargs, task_kwargs: task_kwargs}
    ],
}

*/

// task_set1: [task1, task2], task_set2: [task3, task4] --> [task1, task2, task3, task4]
local flatten_task_sets(task_sets) = std.flatMap(
    function(task_set) std.map(
        function(task) {
            task_set: task_set.name,
            task_name: task.task_name,
            prediction_kwargs: task.prediction_kwargs,
            task_kwargs: task.task_kwargs
        },
        task_set.tasks
    ),
    task_sets
);

local model_task_cross_product(models, task_configs) = std.flatMap(
    function(task_config) std.map(
        function(model_config) model_config + task_config,
        models
    ),
    task_configs
);


local model_task_set_cross_product(models, task_sets) = std.flatMap(
    function(task_set) std.map(
        function(model) {
            task_set: task_set.name,
            model: model.model_path
        },
        models
    ),
    task_sets
);


local basepath(path) =
  // -1 index does not work, so we do this.
  local temp = std.split(path, "/");
  temp[std.length(temp)-1];

local model_location_step_name(model_config) = "model_location_" + basepath(model_config.model_path);
local model_location_ref(model_config) = {type: "ref", ref: model_location_step_name(model_config)};

local create_model_location_steps(models) = std.foldl(
    function(x, model_config) x + {
        [model_location_step_name(model_config)]: {
            type: "get-model-path",
            model_path: model_config.model_path,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    models,
    {}
);


local catwalk_model_step_name(model_config) = "catwalk_model_" + basepath(model_config.model_path);
local catwalk_model_ref(model_config) = {type: "ref", ref: catwalk_model_step_name(model_config)};

local create_catwalk_model_steps(models) = std.foldl(
    function(x, model_config) x + {
        [catwalk_model_step_name(model_config)]: {
            type: "construct-catwalk-model",
            model_path: model_location_ref(model_config),
            model_class: std.get(model_config, "hf_model_class"),
            step_resources: {
                gpu_count: 0
            }
        }
    },
    models,
    {}
);



local task_step_name(config) = "task_" + config.task_set + "_" + config.task_name + std.get(config.task_kwargs, "task_rename", "");
local task_ref(config) = {type: "ref", ref: task_step_name(config)};


local create_task_steps(task_configs) = std.foldl(
    function(x, config) x + {
        [task_step_name(config)]: config.task_kwargs + {
            type: "construct-task",
            task_name: config.task_name,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    task_configs,
    {}
);



local outputs_step_name(config) =
    "outputs_" +
    basepath(config.model_path) + "_" +
    config.task_set + "_" +
    config.task_name + std.get(config.task_kwargs, "task_rename", "");

local outputs_ref(config) = {type: "ref", ref: outputs_step_name(config)};

local create_outputs_steps(model_task_configs) = std.foldl(
    function(x, config) x + {
        [outputs_step_name(config)]: {
            type: "predict-and-calculate-metrics",
            model: catwalk_model_ref(config),
            task_dict: {type: "ref", ref: task_step_name(config)},
            step_resources: {
                gpu_count: config.gpus_needed
            }
        } + config.prediction_kwargs,

    },
    model_task_configs,
    {}
);


local post_process_task_set_step_name(model_path, task_set) =
    "post_process_task_set_" +
    basepath(model_path) + "_" +
    task_set;

local post_process_task_set_ref(model_path, task_set) = {type: "ref", ref: post_process_task_set_step_name(model_path, task_set)};

local all_outputs(task_set, model, model_task_configs) = [
    outputs_ref(config)
    for config in model_task_configs
    if config.task_set == task_set && config.model_path == model
];

local create_post_process_task_set_steps(model_task_sets, model_task_configs) = std.foldl(
    function(x, model_task_set) x + {
        [post_process_task_set_step_name(model_task_set.model, model_task_set.task_set)]: {
            type: "post-process-outputs",
            outputs: all_outputs(model_task_set.task_set, model_task_set.model, model_task_configs),
            model: model_task_set.model,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    model_task_sets,
    {}
);


local create_pipeline(models, task_sets) =

    // Model steps
    local model_location_steps = create_model_location_steps(models);
    local catwalk_model_steps = create_catwalk_model_steps(models);

    // Task steps
    local task_configs = flatten_task_sets(task_sets);
    local task_steps = create_task_steps(task_configs);

    // Prediction and metrics
    local model_task_configs = model_task_cross_product(models, task_configs);
    local outputs_steps = create_outputs_steps(model_task_configs);

    // Aggregate results for each task set and model combination
    //local model_task_sets = model_task_set_cross_product(models, task_sets);
    //local post_process_task_set_steps = create_post_process_task_set_steps(model_task_sets, model_task_configs);

    local all_steps =
        model_location_steps +
        catwalk_model_steps +
        task_steps +
        outputs_steps; // +
        //post_process_task_set_steps;

    all_steps;


{

    create_pipeline: create_pipeline
}

/*local wandb_log_step = {
    logged_metrics: {
        type: "log-metrics",
            //project: "wandb-eval-test",
            //entity: "ai2-llm",
            model_name: "test-olmo-model",
            task_set: "rc20tasks",
            task: "boolq",
            metrics: {type: "ref", "ref": "metrics_test-olmo-model_rc20tasks_boolq"}

    }
};*/