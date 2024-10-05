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
    # main_v0
    # main_v0(args)

    # v0.2
    type_ = "domain"
    if type_ == "domain":
        '''domains_to_write = [
                "Science,_Math_n_Technology", # 29.2M tokens
                "Education_n_Jobs", # 12.4M tokens
                ["History_n_Geography", "Travel_n_Tourism"], # 15.0M tokens
                "Health", # 27.4M tokens
                ["Entertainment", "Art_n_Design", "Social_Life"], # 52.5M tokens
        ]
        max_duration_to_write = ["30e9", "15e9", "15e9", "30e9", "50e9"]
        file_names = ["scitech", "edu", "history", "health", "entertainment"]
        '''
        domains_to_write = [
            ["Adult_Content", "Social_Life", "Literature", "Religion"], # 64.7M tokens literature
            ["History_n_Geography", "Travel_n_Tourism"], # 15.0M tokens history
            ["Art_n_Design", "Entertainment", "Games", "Sports_n_Fitness"], # 64.8M tokens entertainment
            ["Software_Development", "Electronics_n_Hardare", "Science,_Math_n_Technology", "Software"], # 57.9M tokens scitect+sw
            ["Crime_n_Law", "Politics"], # 54.3M tokens politics
            ["Education_n_Jobs"], # 12.4M tokens
            ["Finance_n_Business"], # 21.4M tokens
            ["Health"], # 27.4M tokens
            ["Industrial", "Transportation", "Fashion_n_Beauty", "Food_n_Dining", "Home_n_Hobbies"]
        ]
        max_duration_to_write = [
            "50e9", "15e9", "50e9", "50e9", "50e9", "12.5e9", "20e9", "27.5e9", "25e9"
        ]
        file_names = [
            "lit", "history", "ent_sports",
            "scitech_sw", "politics", "edu",
            "fin", "health", "others"
        ] 
    elif type_ == "format":
        domains_to_Write = [
                ["Structured_Data", "Content_Listing", "Listicle", "Incomplete_Content"],
                ["FAQs", "QnA_Forum", "Interview", "Discussion_Forum", "Personal_About_Page"],
                ["Academic_Writing", "Knowledge_Article", "Organizational_About_Page"],
        ]
        max_duration_to_write = []
        file_names = ["structured", "qna", "knowledge"]
    else:
        raise NotImplementedError()
    assert len(domains_to_write)==len(max_duration_to_write)==len(file_names)
    stop_at_to_write = [str(round(float(n) / (1024 * 4096)) + 10) for n in max_duration_to_write]
      
    postfix = "-from-1T"
    default_config_file = f"sewon-configs/dclm/peteish7-anneal{postfix}-dclmx1.yaml"
    with open(default_config_file, "r") as f:
        config_text = f.read()
    assert config_text.count("paths:")==1
    default_config_text = config_text.split("paths:")[0] + "paths:\n"

    base_dir = "s3://ai2-llm/pretraining-data/sources/ds-olmo-data/dclm_partitioned_v0.2"
    out_base_dir = "/weka/oe-training-default/ai2-llm/preprocessed/categorized-dclm/v0.2_domains"
    
    domain_name_to_paths = {}
    for domain_dir in glob_path(os.path.join(base_dir, "low0", f"{type_}s_v3.8_tokens")):
        domain_name = domain_dir.split("/")[-1] 
        try:
            token_paths1 = list(glob_path(os.path.join(domain_dir, "*.npy")))
            assert len(token_paths1)==16
            token_paths2 = list(glob_path(os.path.join(domain_dir.replace("low0", "high0"), "*.npy")))
            assert len(token_paths2)==32
        except Exception:
            print (f"{domain_name} not done, skip...")
        token_paths = token_paths1 + token_paths2
        token_paths = [token_path.replace(base_dir, out_base_dir) for token_path in token_paths]
        domain_name_to_paths[domain_name] = token_paths

    for domains, max_duration, stop_at, file_name in zip(domains_to_write, max_duration_to_write, stop_at_to_write, file_names):
        if type(domains)==str:
            domains = [domains]
        # file_name = domains[0].replace(",", "").lower()
        curr_config_text = default_config_text
        assert curr_config_text.count("dclmx1")==1
        assert curr_config_text.count("50e9T")==1
        assert curr_config_text.count("11931")==1
        curr_config_text = curr_config_text.replace("dclmx1", file_name)
        curr_config_text = curr_config_text.replace("50e9T", max_duration+"T")
        curr_config_text = curr_config_text.replace("11931", stop_at)

        config_path = f"sewon-configs/dclm_v0.2/peteish7-anneal{postfix}-{file_name}.yaml"
        with open(config_path, "w") as f:
            f.write(curr_config_text)
            for domain in domains:
                f.write(f"    # {domain}\n")
                for token_path in domain_name_to_paths[domain]:
                    f.write(f"    - {token_path}\n")
                f.write("\n\n")

def main_v0(args):
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
    default_config_text = default_config_text.replace("peteish7-anneal-dclmx1", f"peteish7-anneal-{args.config}")

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
    parser.add_argument('-c', '--config', type=str, default=None)
    parser.add_argument('--prefix', type=str, default="peteish7-anneal")
    args = parser.parse_args()
    main(args)



