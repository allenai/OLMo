
local task_utils = import 'task_utils.libsonnet';

local common_kwargs = {
    task_name: "ppl_custom",
    task_kwargs: {
        keep_instance_fields: ["orig_file_name", "source", "subdomain"],
    },
    prediction_kwargs: {
        split: "validation",
        model_max_length: task_utils.model_max_length,
    }
};

// TODO: refactor catwalk's Perplexity task so that it actually uses the s3 path.
// until then, let the path be present in nfs ($EVAL_DATA_PATH).
local data_dir = "olmo-ppl-val-v2-small/";

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
    ["4chan", "c4_100_domains", "c4_en", "gab", "ice", "m2d2_s2orc", "m2d2_wiki",
    "manosphere", "mc4_en", "pile", "ptb", "twitterAEE", "wikitext_103"]
);

{
    task_set: task_utils.create_task_set_from_task_dicts("eval_suite", task_dicts, common_kwargs)
}
