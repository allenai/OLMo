/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local rc20tasks = import 'task_sets/rc20tasks.libsonnet';
local gentasks = import 'task_sets/gentasks.libsonnet';
local ppl_suite = import 'task_sets/eval_suite_ppl_val_v2_small.libsonnet';

// Models to evaluate

local models = [
    {
        model_path: "test_fixtures/test-olmo-model", //"s3://ai2-llm/test_fixtures/olmo-1b"
        gpus_needed: 1
    },
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 1
    }
];

local task_sets = [
    rc20tasks.task_set,
    gentasks.task_set,
    ppl_suite.task_set
];


{
    steps: utils.create_pipeline(models, task_sets)
}
