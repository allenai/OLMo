/*--------------------------------------- Configurations -----------------------------------------*/

local utils = import 'utils.libsonnet';

// These are using smaller test sets.
// ❗Please use full sets 'task_sets/rc20_tasks.libsonnet', etc for your actual experiments.
local rc20_tasks = import 'task_sets/test_sets/test_rc20_tasks.libsonnet';
local gen_tasks = import 'task_sets/test_sets/test_gen_tasks.libsonnet';
local ppl_suite = import 'task_sets/test_sets/test_eval_suite_ppl_val_v2_small.libsonnet';


//❗Set gsheet to the name of your google sheet.
// Set it to null if you do not want your results to be uploaded to a google sheet (they will still be saved as an object).
//local gsheet = "auto-gsheet-test";
local gsheet = null;

// Models to evaluate

local models = [
    {
        model_path: "test_fixtures/test-olmo-model", //❗ Provide remote path: "s3://ai2-llm/test_fixtures/olmo-1b"
        gpus_needed: 1,
        trust_remote_code: true
    },
    {
        model_path: "sshleifer/tiny-gpt2",
        gpus_needed: 1,
        //❗Task sets contain default values for prediction_kwargs. These can be overriden for each model here.
        prediction_kwargs: {
            model_max_length: 128,
            // max_batch_tokens: 32,
            // limit: 1000,
            //fewshot_seed: 1234, //etc.
        }
    },
    /*{
        model_path: "EleutherAI/pythia-1b",
        revision: "step140000", //❗Specify checkpoint if needed
        gpus_needed: 1
    }*/
];

local task_sets = [
    rc20_tasks.task_set,
    gen_tasks.task_set,
    ppl_suite.task_set
];


{
    steps: utils.create_pipeline(models, task_sets, gsheet)
}
