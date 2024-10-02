import os
import argparse
import glob
from tqdm import tqdm
from ipdb import set_trace as bp
# from dolma.core.paths import glob_path
from dolma.core.paths import glob_path

s3_dir = "s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_partitioned_v0.2"
weka_dir = "/weka/oe-training-default/ai2-llm/preprocessed/categorized-dclm/v0.2"

def main(args):
    config_file = args.config_file

    # read original config
    with open(config_file, "r") as f:
        config_text = f.read()
        assert config_text.count("paths:")==1
    
    config_text = config_text.split("paths:")[0] + "paths:\n"

    # write first part
    with open(config_file, "w") as f:
        f.write(config_text)

        paths = []    
        # # write paths: high risk
        for path in tqdm(glob_path(f"{s3_dir}/high0/tokens?/*.npy")):
            path = path.replace(s3_dir, weka_dir)
            paths.append(path)

        # write paths: low risk
        for path in tqdm(glob_path(f"{s3_dir}/low?/tokens/*.npy")):
            # print(path)
            path = path.replace(s3_dir, weka_dir)
            paths.append(path)

        for path in paths:
            f.write(f"    - {path}\n")
        f.write("\n")





if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, required=True)
    parser.add_argument('--copy-from', type=str, default=None)

    parser.add_argument('--no-paywall', action="store_true")
    parser.add_argument('--no-tos', action="store_true")

    args = parser.parse_args()
    main(args)