{
    "steps": {
        "concat": {
            "type" : "concat_strings",
            "string1": "Hello",
            "string2": "World!"
        },
        "hf-env": {
            "type": "env-location-check",
            "env_var": "HF_DATASETS_CACHE"
        }
    }
}