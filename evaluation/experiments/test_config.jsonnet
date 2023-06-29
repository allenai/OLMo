{
    steps: {
        model_location: {
            type: "get-model",
            //model_path: "s3://ai2-llm/test_fixtures/olmo-1b"
            model_path: "test_fixtures/test-olmo-model"
        },
        catwalk_model: {
            type: "construct-catwalk-model",
            model: "lm::pretrained=olmo-1b",
            model_path: {type: "ref", ref: "model_location"},
            model_class: "hf_olmo.OLMoForCausalLM"
        },
        task: {
            type: "construct-task",
            task_name: "boolq"
        },
        predictions: {
            type: "simple-predict",
            model: {type: "ref", ref: "catwalk_model"},
            task: {type: "ref", ref: "task"},
            split: "validation",
            limit: 1000,
            num_shots: 0,
            num_recorded_inputs: 3
        },
        metrics_preds: {
            type: "simple-calculate-metrics",
            model: {type: "ref", ref: "catwalk_model"},
            task: {type: "ref", ref: "task"},
            predictions: {type: "ref", ref: "predictions"}
        }
    }
}