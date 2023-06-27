
# Evaluation

We use tango and catwalk to build the pipeline.

TODO

* beaker_config.yaml (for gantry runs)
* tango workspace (can be gs://... )

- [ ] GetModelStep
- [ ] Catwalk steps (with any updates from olmo-eval branch)
- [ ] SaveResultsStep
- [ ] VisualizeResultsStep (using streamlit.app: can reports be saved as artifacts? If not, have separate command for running the server, pointing to run name).

* New experiments should be run by copying and updating the default config.jsonnet
* Command: `tango --settings beaker_config.yaml run eval_26_june_2023.jsonnet -w gs://llm/eval-workspace`

