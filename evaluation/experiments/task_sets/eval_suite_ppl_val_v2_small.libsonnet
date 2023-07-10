
local task_utils = import 'task_utils.libsonnet';

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
local data_dir = "olmo-ppl-val-v2-small/";

local task_dicts = [
    {
        task_kwargs: {
            task_rename: "ppl_4chan_small",
            files: [data_dir + "4chan/val"]
        },
        prediction_kwargs: {

        }
    },
    {
        task_kwargs: {
            task_rename: "ppl_c4_100_domains_small",
            files: [data_dir + "c4_100_domains/val"]
        },
        prediction_kwargs: {

        }
    }
];

{
    task_set: task_utils.create_task_set_from_task_dicts("eval_suite", task_dicts, common_kwargs)
}