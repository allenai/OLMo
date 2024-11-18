import json
import matplotlib.pyplot as plt
import numpy as np

MODELS = [
    'allenai/OLMo-7B-0724-hf',
    # 'allenai/OLMo-1B-0724-hf',
    # 'allenai/OLMo-7B-0424-hf',
    'allenai/OLMo-7B-hf',
    'allenai/OLMo-1B-hf',
    'meta-llama/Llama-3.2-3B',
    'meta-llama/Llama-3.2-1B',
    # 'meta-llama/Llama-3.1-70B',
    'meta-llama/Llama-3.1-8B',
    # 'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Meta-Llama-3-8B',
    # 'meta-llama/Llama-2-70b-hf',
    # 'meta-llama/Llama-2-13b-hf',
    # 'meta-llama/Llama-2-7b-hf',
    # 'google/gemma-2-27b',
    # 'google/gemma-2-9b',
    # 'google/gemma-2-2b',
    # 'google/gemma-7b',
    # 'google/gemma-2b',
    # 'Qwen/Qwen2.5-72B',
    # 'Qwen/Qwen2.5-32B',
    'Qwen/Qwen2.5-14B',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-3B',
    'Qwen/Qwen2.5-1.5B',
    # 'Qwen/Qwen2-72B',
    'Qwen/Qwen2-7B',
    'Qwen/Qwen2-1.5B',
    'mistralai/Mistral-Nemo-Base-2407',
    'mistralai/Mistral-7B-v0.3',
    'mistralai/Mistral-7B-v0.1',
]

COLOR_BY_MODEL_PREFIX = {
    'allenai': 'hotpink',
    'meta-llama/Llama-3.2': 'darkblue',
    'meta-llama/Llama-3.1': 'mediumblue',
    'meta-llama/Meta-Llama-3': 'royalblue',
    'meta-llama/Llama-2': 'cornflowerblue',
    'google/gemma-2-': 'darkgreen',
    'google/gemma-': 'forestgreen',
    'Qwen/Qwen2.5': 'darkviolet',
    'Qwen/Qwen2': 'violet',
    'mistralai': 'darkorange',
}
def get_color(model):
    for prefix, color in COLOR_BY_MODEL_PREFIX.items():
        if model.startswith(prefix):
            return color
    return 'black'

METRICS_BY_TASK = {
    'rc_rc_mmlu': [
        ('mmlu_stem_var_bpb', 'mmlu_stem_var_len_norm', 0.215),
        ('mmlu_humanities_var_bpb', 'mmlu_humanities_var_len_norm', 0.335),
        ('mmlu_social_sciences_var_bpb', 'mmlu_social_sciences_var_len_norm', 0.219),
        ('mmlu_other_var_bpb', 'mmlu_other_var_len_norm', 0.231),
    ],
    'rc_rc_hellaswag': [('hellaswag_rc_5shot_bpb', 'hellaswag_rc_5shot_len_norm', 1.0)],
    'rc_rc_arc-c': [('arc_challenge_rc_5shot_bpb', 'arc_challenge_rc_5shot_len_norm', 1.0)],
    'rc_rc_piqa': [('piqa_rc_5shot_bpb', 'piqa_rc_5shot_len_norm', 1.0)],
    'rc_rc_csqa': [('csqa_rc_5shot_bpb', 'csqa_rc_5shot_len_norm', 1.0)],
    'rc_rc_socialiqa': [('socialiqa_rc_5shot_bpb', 'socialiqa_rc_5shot_len_norm', 1.0)],
    'rc_mc_mmlu': [
        ('mmlu_stem_var_bpb', 'mmlu_stem_mc_5shot_len_norm', 0.215),
        ('mmlu_humanities_var_bpb', 'mmlu_humanities_mc_5shot_len_norm', 0.335),
        ('mmlu_social_sciences_var_bpb', 'mmlu_social_sciences_mc_5shot_len_norm', 0.219),
        ('mmlu_other_var_bpb', 'mmlu_other_mc_5shot_len_norm', 0.231),
    ],
    'rc_mc_hellaswag': [('hellaswag_rc_5shot_bpb', 'hellaswag_mc_5shot_acc', 1.0)],
    'rc_mc_arc-c': [('arc_challenge_rc_5shot_bpb', 'arc_challenge_mc_5shot_acc', 1.0)],
    'rc_mc_piqa': [('piqa_rc_5shot_bpb', 'piqa_mc_5shot_acc', 1.0)],
    'rc_mc_csqa': [('csqa_rc_5shot_bpb', 'csqa_mc_5shot_acc', 1.0)],
    'rc_mc_socialiqa': [('socialiqa_rc_5shot_bpb', 'socialiqa_mc_5shot_acc', 1.0)],
    'mc_mc_mmlu': [
        ('mmlu_stem_mc_5shot_bpb', 'mmlu_stem_mc_5shot_len_norm', 0.215),
        ('mmlu_humanities_mc_5shot_bpb', 'mmlu_humanities_mc_5shot_len_norm', 0.335),
        ('mmlu_social_sciences_mc_5shot_bpb', 'mmlu_social_sciences_mc_5shot_len_norm', 0.219),
        ('mmlu_other_mc_5shot_bpb', 'mmlu_other_mc_5shot_len_norm', 0.231),
    ],
    'mc_mc_hellaswag': [('hellaswag_mc_5shot_bpb', 'hellaswag_mc_5shot_acc', 1.0)],
    'mc_mc_arc-c': [('arc_challenge_mc_5shot_bpb', 'arc_challenge_mc_5shot_acc', 1.0)],
    'mc_mc_piqa': [('piqa_mc_5shot_bpb', 'piqa_mc_5shot_acc', 1.0)],
    'mc_mc_csqa': [('csqa_mc_5shot_bpb', 'csqa_mc_5shot_acc', 1.0)],
    'mc_mc_socialiqa': [('socialiqa_mc_5shot_bpb', 'socialiqa_mc_5shot_acc', 1.0)],
}

fig, axs = plt.subplots(6, 3, figsize=(3 * 6, 6 * 4.5))

for i, (task, metrics) in enumerate(METRICS_BY_TASK.items()):
    ax = axs[i % 6, i // 6]
    for model in MODELS:
        with open(f'wandb/eval_bpb_mc/{model.replace("/", "_")}.json') as f:
            data = json.load(f)
        try:
            rc_bpb = np.average([data[f'eval/downstream_bpb/{metric[0]}_bpb'] for metric in metrics], weights=[metric[2] for metric in metrics])
            acc = np.average([data[f'eval/downstream/{metric[1]}'] for metric in metrics], weights=[metric[2] for metric in metrics])
        except KeyError:
            continue
        color = get_color(model)
        ax.scatter([rc_bpb], [acc], color=color, s=100)
        ax.annotate(
            text=model.split('/')[1],
            xy=(float(rc_bpb), float(acc)),
            xytext=(8, -3),
            textcoords='offset points',
            fontsize=8,
        )
    ax.set_xlabel(f'{task.split("_")[0]} bpb')
    ax.set_ylabel(f'{task.split("_")[1]} acc')
    ax.set_title(task)

plt.savefig(f'wandb/eval_bpb_mc/all.png', dpi=300, bbox_inches='tight')
