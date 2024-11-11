import os
import argparse

from collections import defaultdict
from dolma.core.paths import glob_path

def read_dclm_data():
    dclm_file = f"sewon-configs/dclm/{args.prefix}-dclmx1.yaml"
    with open(dclm_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    dclm_data_text = config_text.split("paths:")[1]
    return dclm_data_text

def read_lb_v0_data():
    # now, get paths
    s3_dir = "s3://ai2-lucas-archival/pretraining-data/sources/libgen/lb_v0_refined_combined_revised/tokens"
    weka_dir = "/weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/lb_v0"

    paths = list(glob_path(os.path.join(s3_dir, "*.npy")))
    paths = [path.replace(s3_dir, weka_dir) for path in paths]
    return paths

def read_fwedu_data():
    s3_dir = "s3://ai2-llm/pretraining-data/sources/ds-olmo-data/fineweb-edu"
    weka_dir = "/weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/fw-edu"

    paths = list(glob_path(os.path.join(s3_dir, "score?-tokens", "*.npy")))
    paths = [path.replace(s3_dir, weka_dir) for path in paths]
    return paths


def main(args):
    default_config_file = f"sewon-configs/dclm/{args.prefix}-from-1T-dclmx1.yaml"
    with open(default_config_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    default_config_text = config_text.split("paths:")[0]
   
    dclm_data_text = read_dclm_data()
    lb_paths = read_lb_v0_data()
    fwedu_paths = read_fwedu_data() 
    
    # out_file = f"sewon-configs/lb/{args.prefix}-lb_v0.yaml"
    # out2_file = f"sewon-configs/lb/{args.prefix}-dclm+lb_v0x35.yaml"
    # write_configs(out_file, out2_file, default_config_text, dclm_data_text, lb_paths)

    out_file = f"sewon-configs/lb/{args.prefix}-fwedu.yaml"
    out2_file = None
    write_configs(out_file, out2_file, default_config_text, dclm_data_text, lb_paths)

def write_configs(out_file, out2_file, default_config_text, dclm_data_text, paths):
    
    def modify_default_config_text(default_config_text, out_file):
        default_config_text = default_config_text.replace("run_name: peteish7-anneal-from-1T-dclmx1", "run_name: " + out_file.split("/")[-1].split(".")[0])
        default_config_text = default_config_text.replace("project: sewonm-peteish7-anneal", "project: spicy-olmo-medium")
        default_config_text = default_config_text.replace("save_folder: /weka/oe-training-default/ai2-llm/checkpoints/sewonm-peteish7-anneal/", "save_folder: /weka/oe-training-default/ai2-llm/checkpoints/Spicy-OLMo/")
        default_config_text += "paths:\n\n"
        return default_config_text

    with open(out_file, "w") as f:
        f.write(modify_default_config_text(default_config_text, out_file))
        for path in paths:
            f.write(f"    - {path}\n")
        f.write("\n\n")

    if out2_file is None:
        return

    with open(out2_file, "w") as f:
        f.write(modify_default_config_text(default_config_text, out2_file))
        f.write(dclm_data_text + "\n\n")

        for _ in range(35):
            for path in paths:
                f.write(f"    - {path}\n")
            f.write("\n\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--prefix', type=str, default="peteish7-anneal")
    args = parser.parse_args()
    main(args)



