/*--------------------------------------- Configurations -----------------------------------------*/

// Models to evaluate

local models = [
    {
        model_path: "test_fixtures/test-olmo-model", //"s3://ai2-llm/test_fixtures/olmo-1b"
        catwalk_wrapper: "lm::pretrained=olmo-1b",
        hf_model_class: "hf_olmo.OLMoForCausalLM",
        gpus_needed: 0
    },
    {
        model_path: "sshleifer/tiny-gpt2",
        catwalk_wrapper: "lm::pretrained=sshleifer/tiny-gpt2",
        gpus_needed: 0
    }
];

// Defaults (can be overridden for specific task sets)
local model_max_length = 256;

local task_set1 = {
    name: "rc20tasks",
    tasks: ["boolq"],
    prediction_kwargs: {
        split: "validation",
        limit: 1000,
        num_shots: 0,
        num_recorded_inputs: 3,
        model_max_length: model_max_length
    }
};

local task_set2 = {
    name: "gentasks",
    tasks: ["drop"],
    prediction_kwargs: {
        split: "validation",
        limit: 1000,
        num_shots: 5,
        fewshot_seed: 1234,
        num_recorded_inputs: 3,
        model_max_length: model_max_length,
    }
};

local task_sets = [
    task_set1,
    task_set2
];


/*------------------------------- Do Not Edit Beyond This ----------------------------------------*/


// Cross product of models and tasks

local task_configs = std.flatMap(
    function(task_set) std.map(
        function(task) {
            task_set: task_set.name,
            task: task,
            prediction_kwargs: task_set.prediction_kwargs
        },
        task_set.tasks
    ),
    task_sets
);

local model_task_configs = std.flatMap(
    function(task_config) std.map(
        function(model_config) model_config + task_config,
        models
    ),
    task_configs
);


/* ------------------------------------------ Model steps ----------------------------------------*/

local basepath(path) =
  // -1 index does not work, so we do this.
  local temp = std.split(path, "/");
  temp[std.length(temp)-1];

local model_location_step_name(model_config) = "model_location_" + basepath(model_config.model_path);
local model_location_ref(model_config) = {type: "ref", ref: model_location_step_name(model_config)};

local model_location_steps = std.foldl(
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

local catwalk_model_steps = std.foldl(
    function(x, model_config) x + {
        [catwalk_model_step_name(model_config)]: {
            type: "construct-catwalk-model",
            model: model_config.catwalk_wrapper,
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

/* ------------------------------------------ Task steps -----------------------------------------*/

local task_step_name(config) = "task_" + config.task_set + "_" + config.task;
local task_ref(config) = {type: "ref", ref: task_step_name(config)};

local task_steps = std.foldl(
    function(x, config) x + {
        [task_step_name(config)]: {
            type: "construct-task",
            task_name: config.task,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    task_configs,
    {}
);

/* ----------------------------- Prediction and metric steps -----------------------------------*/

local outputs_step_name(config) =
    "outputs_" +
    basepath(config.model_path) + "_" +
    config.task_set + "_" +
    config.task;

local outputs_ref(config) = {type: "ref", ref: outputs_step_name(config)};

local outputs_steps = std.foldl(
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


/* --------------------------------------- Postprocess steps -------------------------------------*/

// Aggregate results for each task set and model combination

local model_task_sets = std.flatMap(
    function(task_set) std.map(
        function(model) {
            task_set: task_set.name,
            model: model.model_path
        },
        models
    ),
    task_sets
);

local post_process_task_set_step_name(model_path, task_set) =
    "post_process_task_set_" +
    basepath(model_path) + "_" +
    task_set;

local post_process_task_set_ref(model_path, task_set) = {type: "ref", ref: post_process_task_set_step_name(model_path, task_set)};

local all_outputs(task_set, model) = [
    outputs_ref(config)
    for config in model_task_configs
    if config.task_set == task_set && config.model_path == model
];

local post_process_task_set_steps = std.foldl(
    function(x, model_task_set) x + {
        [post_process_task_set_step_name(model_task_set.model, model_task_set.task_set)]: {
            type: "post-process-outputs",
            outputs: all_outputs(model_task_set.task_set, model_task_set.model),
            model: model_task_set.model,
            step_resources: {
                gpu_count: 0
            }
        }
    },
    model_task_sets,
    {}
);

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

/* ----------------------------------------- All steps ----------------------------------------*/

{
    steps:
        model_location_steps +
        catwalk_model_steps +
        task_steps +
        outputs_steps +
        post_process_task_set_steps
}

//TODO: put all step creations and functions into a utils.libsonnet.
//experiment files should only have the model and task configs.