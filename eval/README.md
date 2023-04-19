# OLMo evaluation

## Downstream task evaluation during pretraining

Initial (hacky) code for connecting OLMo to Catwalk. For now, clone Catwalk fork from 
https://github.com/OyvindTafjord/catwalk and install into your OLMo environment with
```commandline
cd /path/to/catwalk
pip install -e .
```

Sample code to run eval with OLMo checkpoint, assumed to be run in OLMo root directory:
```commandline
python -m eval.catwalk_eval --model rc::pretrained=300m-c4,revision=78000 \ 
  --split validation --task arc_easy --limit 10 \
  --olmo_model_path models/ed5krfk9 --full_output_file tmpoutput.jsonl \
  --metrics_file metrics.json -d ./my-workspace --num_model_inputs 2
```

See [this document]() for more information.

