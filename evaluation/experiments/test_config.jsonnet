/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

local rc20tasks = import 'task_sets/rc20tasks.libsonnet';
local gentasks = import 'task_sets/gentasks.libsonnet';

// Models to evaluate

local models = [
    {
        model_path: "test_fixtures/test-olmo-model", //"s3://ai2-llm/test_fixtures/olmo-1b"
        hf_model_class: "hf_olmo.OLMoForCausalLM",
        gpus_needed: 0
    },
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 0
    }
];

local task_sets = [
    rc20tasks.task_set,
    gentasks.task_set
];

{
    steps: utils.create_pipeline(models, task_sets)
}
