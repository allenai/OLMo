import os
import argparse

from collections import defaultdict
from dolma.core.paths import glob_path

def find_domain(line):
    if "proof-pile-2/" in line:
        return "math"
    elif "pes2o/" in line:
        return "pes2o"
    elif "starcoder/" in line:
        return "starcoder"
    elif "dclm/text_openhermes_reddit_eli5_vs" in line:
        return "dclm"
    elif "documents/wiki/" in line:
        return "wiki"
    elif "Spicy-OLMo/books3/" in line:
        return "books"
    else:
        raise NotImplementedError()

def main(args):
    default_config_file = f"sewon-configs/dclm/{args.prefix}-from-1T-dclmx1.yaml"
    dclm_file = f"sewon-configs/dclm/{args.prefix}-dclmx1.yaml"
    out_file = f"sewon-configs/lb/{args.prefix}-lb_v0.yaml"
    out2_file = f"sewon-configs/lb/{args.prefix}-dclm+lb_v0x35.yaml"
    
    with open(default_config_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    default_config_text = config_text.split("paths:")[0]
   
    with open(dclm_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    dclm_data_text = config_text.split("paths:")[1]

    # now, get paths
    s3_dir = "s3://ai2-lucas-archival/pretraining-data/sources/libgen/lb_v0_refined_combined_revised/tokens"
    weka_dir = "/weka/oe-training-default/ai2-llm/preprocessed/Spicy-OLMo/lb_v0"

    paths = list(glob_path(os.path.join(s3_dir, "*.npy")))
    paths = [path.replace(s3_dir, weka_dir) for path in paths]

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



