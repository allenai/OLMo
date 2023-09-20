# Preprocessing ICE with HELMs

In order to make our use of the ICE dataset comprable to HELMs, we use their code to preprocess the data.

## setup

```
conda create -n crfm-helm python=3.10
conda activate crfm-helm
pip install -r requirements.txt
```

## preprocess ICE

First we need to put the dataset in the dir `restricted` in this directory. This assumes you've obtained access to the ICE datasets and used the associed passcodes to unzip them. Make sure to include the ICE-IRELAND subset instead of the SPICE-IRL subset, as the latter is not used by HELM. Internal AI2 users can find the raw data in the data bucket.

Then run the preprocessing script
```
python dump_ice.py --out_dir <out_dir>
```

The files in the out_dir will now be ready for processing by `LLM/eval_data/format_conversion/eval_data_converter.py`