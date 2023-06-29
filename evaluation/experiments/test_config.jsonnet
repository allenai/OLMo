/*--------------------------------------- Configurations -----------------------------------------*/

// Models to evaluate

local model_paths = ["test_fixtures/test-olmo-model"];  //"s3://ai2-llm/test_fixtures/olmo-1b"

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
        function(model_path) {
            model_path: model_path,
        } + task_config,
        model_paths
    ),
    task_configs
);



/* ------------------------------------------ Model steps ----------------------------------------*/

//TODO: it should be last index [-1], but for some reason, it does not work.
local index = 1;

local model_location_step_name(model_path) = "model_location_" + std.split(model_path, "/")[index];
local model_location_ref(model_path) = {type: "ref", ref: model_location_step_name(model_path)};

//TODO: specify step_resources

local model_location_steps = std.foldl(
    function(x, model_path) x + {
        [model_location_step_name(model_path)]: {
            type: "get-model",
            model_path: model_path,
        }
    },
    model_paths,
    {}
);

local catwalk_model_step_name(model_path) = "catwalk_model_" + std.split(model_path, "/")[index];
local catwalk_model_ref(model_path) = {type: "ref", ref: catwalk_model_step_name(model_path)};

local catwalk_model_steps = std.foldl(
    function(x, model_path) x + {
        [catwalk_model_step_name(model_path)]: {
            type: "construct-catwalk-model",
            model: "lm::pretrained=olmo-1b", //TODO: use configs instead, to allow for other types
            model_path: model_location_ref(model_path),
            model_class: "hf_olmo.OLMoForCausalLM" //TODO
        }
    },
    model_paths,
    {}
);

/* ------------------------------------------ Task steps -----------------------------------------*/

local task_step_name(config) = "task_" + config.task_set + "_" + config.task;
local task_ref(model_path) = {type: "ref", ref: task_step_name(config)};

local task_steps = std.foldl(
    function(x, config) x + {
        [task_step_name(config)]: {
            type: "construct-task",
            task_name: config.task,
        }
    },
    task_configs,
    {}
);


/* --------------------------------------- Prediction steps -------------------------------------*/


local predictions_step_name(config) =
    "predictions_" +
    std.split(config.model_path, "/")[index] + "_" +
    config.task_set + "_" +
    config.task;

local predictions_ref(config) = {type: "ref", ref: predictions_step_name(config)};

local predictions_steps = std.foldl(
    function(x, config) x + {
        [predictions_step_name(config)]: {
            type: "simple-predict",
            model: catwalk_model_ref(config.model_path),
            task: {type: "ref", ref: task_step_name(config)},
        } + config.prediction_kwargs,
    },
    model_task_configs,
    {}
);

/* ----------------------------------------- Metrics steps ----------------------------------------*/

local metrics_step_name(config) =
    "metrics_" +
    std.split(config.model_path, "/")[index] + "_" +
    config.task_set + "_" +
    config.task;

local metrics_ref(config) = {type: "ref", ref: metrics_step_name(config)};

local metrics_steps = std.foldl(
    function(x, config) x + {
        [metrics_step_name(config)]: {
            type: "simple-calculate-metrics",
            model: catwalk_model_ref(config.model_path),
            task: {type: "ref", ref: task_step_name(config)},
            predictions: predictions_ref(config)
        }
    },
    model_task_configs,
    {}
);

local wandb_log_step = {
    logged_metrics: {
        type: "log-metrics",
            //project: "wandb-eval-test",
            //entity: "ai2-llm",
            model_name: "test-olmo-model",
            task_set: "rc20tasks",
            task: "boolq",
            metrics: {type: "ref", "ref": "metrics_test-olmo-model_rc20tasks_boolq"}

    }
};

/* ----------------------------------------- All steps ----------------------------------------*/

{
    steps:
        model_location_steps +
        catwalk_model_steps +
        task_steps +
        predictions_steps +
        metrics_steps +
        wandb_log_step
}
