import wandb
import json

api = wandb.Api()
run = api.run('ai2-llm/OLMo-2-1124-7B/7xkf4smi')

config = run._attrs["rawconfig"]
data_paths = config["data"]["paths"]
data_paths = [data_path.replace("s3://", "/weka/oe-training-default/") for data_path in data_paths]

with open('datapaths_dolmino-mix-1124-7b-50b.json', 'w') as f:
    json.dump(data_paths, f, indent=4)
