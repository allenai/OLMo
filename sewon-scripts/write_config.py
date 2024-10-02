import argparse
from collections import defaultdict

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
    # get config other than paths
    default_config_file = f"sewon-configs/dclm/{args.prefix}-dclmx1.yaml"
    with open(default_config_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    default_config_text = config_text.split("paths:")[0] + "paths:\n"

    # now, get paths
    with open("sewon-configs/peteish7-anneal-B3x50-weka.yaml", "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    path_text = config_text.split("paths:")[1]
    paths = [line for line in path_text.split("\n") if len(line.strip())>0 and not line.strip().startswith("#")]
    
    domain2paths = defaultdict(list)
    for path in set(paths):
        domain2paths[find_domain(path)].append(path)

    # args.config should be in the format of "dclmx1_starcoderx20"
    # 1) dclmx1
    # 2) dclmx1_codex20
    # 3) dclmx1_booksx50
    # 4) dclmx1_mathx30

    out_path = f"sewon-configs/dclm/{args.prefix}-{args.config}.yaml"
    default_config_text = default_config_text.replace("peteish-anneal-dclmx1", f"peteish-anneal-{args.config}")

    with open(out_path, "w") as f:
        f.write(default_config_text)

        for c in args.config.split("_"):
            domain, multiplier = c.split("x")
            paths = []
            for _ in range(int(multiplier)):
                paths += domain2paths[domain]
            f.write(f"    #### {domain} (x{multiplier})\n")
            for path in paths:
                f.write(path + "\n")
            f.write("\n\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--prefix', type=str, default="peteish7-anneal")
    args = parser.parse_args()
    main(args)



