mixtures = [
    "dolma17",
    "no_math_no_code",
    "no_reddit",
    "no_flan",
    "no_code",
    "falcon",
    "c4",
    "falcon_and_cc",
    "falcon_and_cc_eli5_oh_top10p",
    "falcon_and_cc_eli5_oh_top20p",
    "falcon_and_cc_og_eli5_oh_top10p",
    "prox_fineweb_pro",
    "fineweb_edu_dedup",
    "falcon_and_cc_tulu_qc_top10",
    "DCLM-baseline",
    "dolma17-75p-DCLM-baseline-25p",
    "dolma17-50p-DCLM-baseline-50p",
    "dolma17-25p-DCLM-baseline-75p",
    "dolma-v1-6-and-sources-baseline",
]

sizes = [
    "150M",
    "300M",
    "530M",
    "750M",
    "1B"
]

seeds = [
    None,
    2,
    3
]

for mixture in mixtures:
    for size in sizes:
        for seed in seeds:
            bash_command = f"bash scripts/beaker/ladder-launch.sh 1 normal --model {size} --data {mixture} --length 5xC --name {mixture} --s3 --save_overwrite"
            if seed is not None:
                bash_command += f" --seed {seed}"
            print(bash_command)