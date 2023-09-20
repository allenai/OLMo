from helm.benchmark.scenarios import ice_scenario
from helm.benchmark.scenarios.ice_scenario import TextCategory
import argparse
import os
import json
from tqdm import tqdm
import gzip

def main(args):
    for i, subset in enumerate(ice_scenario.ICESubset):
        for category in [TextCategory.W_ALL, TextCategory.S_ALL]:
            with gzip.open(os.path.join(args.out_dir,f"{subset.name}_{category.name}.jsonl.gz"), 'wt') as f:
                ice = ice_scenario.ICEScenario(subset.value, category=category)
                for instance in tqdm(ice.get_instances(), desc=f"Dumping {subset.name} ({i+1}/{len(ice_scenario.ICESubset)})"):
                    json_instance = {
                        'text':instance.input.text,
                        'subdomain':f'{subset.name}_{category.name}',
                    }
                    f.write(json.dumps(json_instance) + '\n')

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--category', type=str, default='all')
    args = parser.parse_args()
    main(args)