from scripts.ladder import parse_size, parse_length, calculate_batch_size


def compute_stop_at(
        percentage_of_target: float,
        current_run_parameters: str,
        target_parameters: str,
        target_multiplier: str,
        sequence_length = 2048
):
    target_parameters_parsed = parse_size(target_parameters)
    target_length_parsed = parse_length(target_multiplier, target_parameters_parsed)
    target_compute = 6 * target_length_parsed * target_parameters_parsed

    current_run_parameters_parsed = parse_size(current_run_parameters)
    current_run_batch_size = calculate_batch_size(sequence_length, current_run_parameters_parsed)

    stop_at_compute = int(percentage_of_target * target_compute)
    tokens_per_step = current_run_batch_size * sequence_length

    stop_at_tokens = stop_at_compute // (current_run_parameters_parsed * 6)
    stop_at_steps = stop_at_tokens // tokens_per_step

    return stop_at_steps


if __name__ == "__main__":
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

    stop_at_configs = [
        # None,
        0.25
    ]

    for mixture in mixtures:
        for size in sizes:
            for seed in seeds:
                for stop_at_config in stop_at_configs:
                    if stop_at_config is not None:
                        stop_at = compute_stop_at(stop_at_config, size, "1B", "5xC")
                        # change seed so that we don't use same seeds as target
                        if seed is None:
                            seed = 0
                        seed += 10
                    else:
                        stop_at = None
                    bash_command = f"bash scripts/beaker/ladder-launch.sh 1 normal --model {size} --data {mixture} --length 5xC --name {mixture} --s3 --save_overwrite"
                    if seed is not None:
                        bash_command += f" --seed {seed}"
                    if stop_at_config is not None:
                        bash_command += f" --stop_at {stop_at}"
                    print(bash_command)