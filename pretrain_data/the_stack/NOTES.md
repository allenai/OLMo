# Overview

[The Stack](https://huggingface.co/datasets/bigcode/the-stack) is a 6 TB dataset of code, containing 358 programming languages. This is too large for our purpose, and research has shown that deduplication improves model performance. 
We use the [deduplicated version](https://huggingface.co/datasets/bigcode/the-stack) of The Stack, which contains 3 TB of data.

**In Progress**: The dataset is being downloaded on lm-datasets.

Sample instance:
```
{'hexsha': '900000adc880b3f11205aecc535df29d629a6d0c',
 'size': 3225,
 'ext': 'py',
 'lang': 'Python',
 'max_stars_repo_path': 'tests/mcts_benchmark.py',
 'max_stars_repo_name': 'jclarte/quarto',
 'max_stars_repo_head_hexsha': 'e3767ec899c9b2cfe3e4c0ecb875d932cb01a27d',
 'max_stars_repo_licenses': ['MIT'],
 'max_stars_count': None,
 'max_stars_repo_stars_event_min_datetime': None,
 'max_stars_repo_stars_event_max_datetime': None,
 'max_issues_repo_path': 'tests/mcts_benchmark.py',
 'max_issues_repo_name': 'jclarte/quarto',
 'max_issues_repo_head_hexsha': 'e3767ec899c9b2cfe3e4c0ecb875d932cb01a27d',
 'max_issues_repo_licenses': ['MIT'],
 'max_issues_count': 2.0,
 'max_issues_repo_issues_event_min_datetime': '2020-06-07T17:30:05.000Z',
 'max_issues_repo_issues_event_max_datetime': '2020-06-09T19:45:21.000Z',
 'max_forks_repo_path': 'tests/mcts_benchmark.py',
 'max_forks_repo_name': 'jclarte/quarto',
 'max_forks_repo_head_hexsha': 'e3767ec899c9b2cfe3e4c0ecb875d932cb01a27d',
 'max_forks_repo_licenses': ['MIT'],
 'max_forks_count': None,
 'max_forks_repo_forks_event_min_datetime': None,
 'max_forks_repo_forks_event_max_datetime': None,
 'content': '"""\nbenchmarking et plotting some results for quarto mcts\n"""\n\nimport time\nimport statistics\nimport sys\nimport os\nimport pytest\n\nimport matplotlib.pyplot as plt\nsys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))\n\nfrom logic.quarto_logic import Quarto\nfrom logic.mcts import MCTS\nfrom logic.mcts_quarto import QuartoInterface\n\n\n\n\ndef plot_time_per_n_iter():\n    times = []\n    for n_iter in range(1, 1001):\n        game = Quarto()\n        bot = MCTS(QuartoInterface)\n        start = time.time()\n        bot.run(game, n_iter)\n        end = time.time()\n        sim_time = (end - start)\n        print(f"n_iter={n_iter}, time={sim_time}")\n        times.append(sim_time)\n    plt.plot(times)\n    plt.show()\n\ndef plot_memory_vs_no_memory():\n    scores = [0, 0, 0]\n    win_ratio = []\n    n_iter = [20, 100]\n    game = Quarto()\n    bot_0 = MCTS(QuartoInterface)\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_0, bot_1][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio.append(scores[0]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio={win_ratio[-1]}")\n    plt.plot(win_ratio)\n    plt.show()\n\ndef plot_memory_vs_no_memory_alter():\n    scores = [0, 0, 0]\n    win_ratio = []\n    n_iter = [20, 100]\n    game = Quarto()\n    bot_0 = MCTS(QuartoInterface)\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_1, bot_0][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio.append(scores[1]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio={win_ratio[-1]}")\n    plt.plot(win_ratio)\n    plt.show()\n\n\ndef plot_win_ratio_convergence():\n    scores = [0, 0, 0]\n    win_ratio_0 = []\n    win_ratio_1 = []\n    tie_ratio = []\n    n_iter = [1000, 1000]\n    game = Quarto()\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_0 = MCTS(QuartoInterface)\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_1, bot_0][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio_0.append(scores[0]/n_sim)\n        win_ratio_1.append(scores[1]/n_sim)\n        tie_ratio.append(scores[2]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio_0={win_ratio_0[-1]:.1%}, win_ratio_1={win_ratio_1[-1]:.1%}, tie_ratio_0={tie_ratio[-1]:.1%}")\n    plt.plot(win_ratio_0)\n    plt.plot(win_ratio_1)\n    plt.plot(tie_ratio)\n    plt.show()\n\nif __name__ == \'__main__\':\n    # plot_time_per_n_iter()\n    # plot_memory_vs_no_memory()\n    # plot_memory_vs_no_memory_alter()\n    plot_win_ratio_convergence()\nelse:\n    raise ImportError("Can\'t import this script")',
 'avg_line_length': 29.8611111111,
 'max_line_length': 135,
 'alphanum_fraction': 0.6096124031}
```


## Download

Download script here (will be copied over): https://github.com/allenai/wimbd/blob/main/wimbd/indexing/huggingface.py

Usage:
```
pip install datasets
python wimbd/indexing/huggingface.py \
  --dataset bigcode/the-stack-dedup \  # any huggingface dataset name
  --output-dir /mnt/tank/the-stack-dedup \
  --batch-size 500000 \  # Number of instances in memory, written to a single file
  --num-workers 10 \  # Number of parallel workers
  --text-field content
```



## Notes / Comments

1. The Stack allows Github users to opt out of being part of the dataset. We timestamp the version we are downloading, is there something else that we should take care of?

2. Notes about the deduplication performed:

  - Remove exact duplicates
  - Remove near duplicates (MinHash + LSH). https://github.com/bigcode-project/bigcode-analysis/tree/main/data_analysis
