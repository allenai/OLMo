from helm.benchmark.scenarios import ice_scenario
import argparse
import os
import json
from tqdm import tqdm

def main(args):
    for i, subset in enumerate(ice_scenario.ICESubset):
        with open(os.path.join(args.out_dir,f"{subset.name}.jsonl"), 'w') as f:
            ice = ice_scenario.ICEScenario(subset.value, category=args.category)
            for instance in tqdm(ice.get_instances(), desc=f"Dumping {subset.name} ({i+1}/{len(ice_scenario.ICESubset)})"):
                json_instance = {'text':instance.input.text}
                f.write(json.dumps(json_instance) + '\n')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--category', type=str, default='all')
    args = parser.parse_args()
    main(args)