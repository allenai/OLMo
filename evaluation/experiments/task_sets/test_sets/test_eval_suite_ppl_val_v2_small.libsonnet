
local task_utils = import '../task_utils.libsonnet';

local task_set_name = "eval_suite";

local common_kwargs = {
    task_name: "ppl_custom",
    task_kwargs: {
        keep_instance_fields: ["orig_file_name", "source", "subdomain"],
    },
    prediction_kwargs: {
        split: "validation",
        model_max_length: 256,
    }
};

// TODO: refactor catwalk's Perplexity task so that it actually uses the s3 path.
// until then, let the path be present in nfs
local data_dir = "test_fixtures/evaluation/ppl-test-data";

local create_task_kwargs(task_names) = [
    {
        task_kwargs: {
            task_rename: "ppl_" + task_name + "_small",
            files: [data_dir + "/" + task_name + "/val"]
        }
    }
    for task_name in task_names
];

local task_dicts = create_task_kwargs(
    ["4chan", "c4_100_domains"]
);

{
    task_set: task_utils.create_task_set_from_task_dicts(task_set_name, task_dicts, common_kwargs)
}