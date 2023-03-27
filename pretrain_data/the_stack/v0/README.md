Author: Akshita Bhagia @akshitab

# Version: v0

[The Stack](https://huggingface.co/datasets/bigcode/the-stack) is a 6 TB dataset of code, containing 358 programming languages. This is too large for our purpose, and research has shown that deduplication improves model performance. 
We use the [deduplicated version](https://huggingface.co/datasets/bigcode/the-stack) of The Stack, which contains 3 TB of data.

[Dataset Statistics](https://docs.google.com/spreadsheets/d/1aesZHgHhmbPLcKvLNdn8dMV8ZVE6ohpIpuXCk6-LUpA/edit#gid=1395653416)

## Downloaded data

Downloaded files available here: `s3://ai2-llm/pretraining-data/sources/stack-dedup/raw` (722 GB compressed)

HuggingFace dataset version: [v1.1](https://huggingface.co/datasets/bigcode/the-stack-dedup/tree/v1.1)

## Steps to Reproduce

### Step1: Setup environment

1. Create conda environment
   ```
   conda create -n llm python=3.10
   conda activate llm
   cd LLM/pretrain_data/the_stack/v0
   pip install -r requirements.txt
   ```

2. Login to HuggingFace on your machine and accept the terms to access the dataset at `https://huggingface.co/datasets/bigcode/the-stack-dedup`.


3. Add HuggingFace API token to the environment.

   ```commandline
   export HF_TOKEN=<YOUR TOKEN>
   ```

4. Configure AWS for copying to S3 bucket
   ```
   aws configure
   ```
   Follow the prompts on the command line.


### Step 2: Download and save on S3 bucket

5. (Optional) Obtain the list of parquet file urls to download.

   **Note**: The list of URLs has already been created here: [stack_dedup_v1.1_urls.txt](stack_dedup_v1.1_urls.txt)
   
   ```commandline
   python get_urls.py --version v1.1 --output-file stack_dedup_v1.1_urls.txt
   ```

6. Run the download script (partial progress will be saved).

   ```commandline
   ./download.sh stack_dedup_v1.1_urls.txt
   ```

### Troubleshooting

* `download.sh` script uses GNU-parallel to download files in parallel. You may see occasional errors where download for some URLs fails. This is likely due to CPU cores having been occupied fully due to processing large files. Rerunning the script should take care of those missing/failed URLs.

## 2023-03-17: First pass

1. Download details:
   - Original: [the-stack-dedup v1.1](https://huggingface.co/datasets/bigcode/the-stack-dedup/tree/v1.1)
   - Downloaded files available here: `s3://ai2-llm/pretraining-data/sources/stack-dedup/raw` (722 GB compressed)
   - 358 language folders, total 4790 gzipped json files.
   - Compute: EC2 `m6a.8xlarge` machine in `us-east-1` region; took < 1 day.


2. Format:
   ```text
      id: Github hexsha of the file commit (original huggingface ID)
      text: content of the file
      lang: programming language
      metadata: all other fields from the dataset (see below for example)
)     timestamp: 2022-12-01 (Release date of dataset v1.1 on HF)
      added: 2023-03-16 / 2023-03-17
      source: stack-dedup
      ```

   Sample instance in our format:
   
   ```
   {'id': '900000adc880b3f11205aecc535df29d629a6d0c',
    'lang': 'python',
    'text': '"""\nbenchmarking et plotting some results for quarto mcts\n"""\n\nimport time\nimport statistics\nimport sys\nimport os\nimport pytest\n\nimport matplotlib.pyplot as plt\nsys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))\n\nfrom logic.quarto_logic import Quarto\nfrom logic.mcts import MCTS\nfrom logic.mcts_quarto import QuartoInterface\n\n\n\n\ndef plot_time_per_n_iter():\n    times = []\n    for n_iter in range(1, 1001):\n        game = Quarto()\n        bot = MCTS(QuartoInterface)\n        start = time.time()\n        bot.run(game, n_iter)\n        end = time.time()\n        sim_time = (end - start)\n        print(f"n_iter={n_iter}, time={sim_time}")\n        times.append(sim_time)\n    plt.plot(times)\n    plt.show()\n\ndef plot_memory_vs_no_memory():\n    scores = [0, 0, 0]\n    win_ratio = []\n    n_iter = [20, 100]\n    game = Quarto()\n    bot_0 = MCTS(QuartoInterface)\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_0, bot_1][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio.append(scores[0]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio={win_ratio[-1]}")\n    plt.plot(win_ratio)\n    plt.show()\n\ndef plot_memory_vs_no_memory_alter():\n    scores = [0, 0, 0]\n    win_ratio = []\n    n_iter = [20, 100]\n    game = Quarto()\n    bot_0 = MCTS(QuartoInterface)\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_1, bot_0][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio.append(scores[1]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio={win_ratio[-1]}")\n    plt.plot(win_ratio)\n    plt.show()\n\n\ndef plot_win_ratio_convergence():\n    scores = [0, 0, 0]\n    win_ratio_0 = []\n    win_ratio_1 = []\n    tie_ratio = []\n    n_iter = [1000, 1000]\n    game = Quarto()\n    for n_sim in range(1, 101):\n        game = Quarto()\n        bot_0 = MCTS(QuartoInterface)\n        bot_1 = MCTS(QuartoInterface)\n        start = time.time()\n        while game.end() == -1:\n            action = [bot_1, bot_0][game.player].run(game, n_iter[game.player])\n            # print(f"Player {game.player} plays {game.state} {action}")\n            game.transition(action)\n        scores[game.end()] += 1\n        win_ratio_0.append(scores[0]/n_sim)\n        win_ratio_1.append(scores[1]/n_sim)\n        tie_ratio.append(scores[2]/n_sim)\n        print(f"n_iter={n_sim}, win_ratio_0={win_ratio_0[-1]:.1%}, win_ratio_1={win_ratio_1[-1]:.1%}, tie_ratio_0={tie_ratio[-1]:.1%}")\n    plt.plot(win_ratio_0)\n    plt.plot(win_ratio_1)\n    plt.plot(tie_ratio)\n    plt.show()\n\nif __name__ == \'__main__\':\n    # plot_time_per_n_iter()\n    # plot_memory_vs_no_memory()\n    # plot_memory_vs_no_memory_alter()\n    plot_win_ratio_convergence()\nelse:\n    raise ImportError("Can\'t import this script")',
    'metadata': {'size': 3225,
     'ext': 'py',
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
     'avg_line_length': 29.8611111111,
     'max_line_length': 135,
     'alphanum_fraction': 0.6096124031},
    'timestamp': Timestamp('2022-12-01 00:00:00'),
    'source': 'stack-dedup',
    'added': '2023-03-17T09:18:30.577245'}
   ```


## 2023-03-27: Dataset Statistics

[Dataset Statistics](https://docs.google.com/spreadsheets/d/1aesZHgHhmbPLcKvLNdn8dMV8ZVE6ohpIpuXCk6-LUpA/edit#gid=1395653416)

Programming languages - 358

Licences - MIT, Apache-2.0, BSD-3-Clause, Unlicense, CC0-1.0, BSD-2-Clause, CC-BY-4.0, CC-BY-3.0, 0BSD, RSA-MD, WTFPL, MIT-0, ISC, ADSL, BSL-1.0, Zlib, Artistic-2.0, FTL, MS-PL, BSD-2-Clause-FreeBSD, FSFAP, BSD-Source-Code, Apache-1.1, BSD-4-Clause, Ruby, Artistic-1.0, MulanPSL-1.0, BSD-1-Clause, X11, CNRI-Python, Beerware, Condor-1.1, PostgreSQL, CECILL-B, Intel, Vim, Naumen, OML, BSD-3-Clause-Clear, AML, PHP-3.01, OpenSSL, PSF-2.0, Xnet, Linux-OpenIB, BSD-3-Clause-LBNL, UPL-1.0, AFL-3.0, BlueOak-1.0.0, Info-ZIP, BSD-4-Clause-UC, AAL, LPPL-1.3c, bzip2-1.0.6, W3C, W3C-20150513, AFL-1.1, DOC, ICU, CC-BY-2.0, curl, MTLL, OLDAP-2.2.1, ECL-2.0, Adobe-Glyph, CNRI-Python-GPL-Compatible, BSD-2-Clause-Patent, IJG, PHP-3.0, ZPL-2.1, MIT-advertising, NCSA, Fair, BSD-3-Clause-Attribution, OLDAP-2.3, NLPL, BSD-3-Clause-Open-MPI, ClArtistic, Python-2.0, NASA-1.3, TCL, Artistic-1.0-Perl, blessing, BSD-3-Clause-No-Nuclear-Warranty, ImageMagick, Net-SNMP, Artistic-1.0-cl8, OLDAP-2.5, MIT-feh, OLDAP-2.4, MITNFA, AFL-2.1, libpng-2.0, EFL-2.0, OLDAP-2.7, IBM-pibs, libtiff, OLDAP-2.8, Cube, Adobe-2006, BSD-2-Clause-NetBSD, zlib-acknowledgement, OLDAP-2.6, BSD-3-Clause-No-Nuclear-License-2014, OLDAP-1.4, Libpng, MIT-CMU, AFL-2.0, JasPer-2.0, LPL-1.02, Zend-2.0, TCP-wrappers, XFree86-1.1, FSFUL, OLDAP-1.3, SGI-B-2.0, NetCDF, CNRI-Jython, Zed, ZPL-2.0, AFL-1.2, Apache-1.0, CC-BY-1.0, OLDAP-2.1, OLDAP-1.2, OLDAP-2.0, NTP, LPL-1.0, AMPAS, Barr, mpich2, ANTLR-PD, Xerox, Spencer-94, AMDPLPA, BSD-3-Clause-No-Nuclear-License, HPND, ECL-1.0, MirOS, Qhull, ZPL-1.1, TU-Berlin-2.0, Spencer-86, SMLNJ, xinetd, OLDAP-2.2.2, OGTSL, MIT-enna, Font-exception-2.0, FSFULLR, TU-Berlin-1.0, xpp, NRL, W3C-19980720, EFL-1.0, eGenix, Unicode-DFS-2016, SWL, Spencer-99, Plexus, VSL-1.0, Leptonica, Unicode-DFS-2015, Mup, Giftware, OLDAP-2.2, APAFML, NBPL-1.0, OLDAP-1.1, Entessa, Multics, Newsletr, psutils, bzip2-1.0.5, Afmparse, diffmark, BSD-2-Clause-Views, DSDP, MIT-Modern-Variant, ANTLR-PD-fallback, Bahyph, BSD-3-Clause-Modification, BSD-4-Clause-Shortened, HTMLTIDY, MIT-open-group, MulanPSL-2.0, OLDAP-2.0.1, Saxpath, Borceux, Crossword, CrystalStacker, Rdisc, Wsuipa

Number of tokens (basic tokenization; spaces and newlines) - 514,363,785,904
Top-10 languages (by number of tokens) - json, html, text, javascript, xml, php, java, python, c++ (364,553,035,516 tokens, approx 70% of total tokens)


### Prior Work

* Gopher

   GitHub:
      Disk size - 3.1 TB
      Number of documents - 142M
      Number of tokens - 422B
      Sampling proportion - 3%


   Sampling strategy: "For Github, we restrict the data to only include code with the following permissive licenses: Apache License version 2.0, MIT license, The 3-clause BSD license, The 2-clause BSD license, Unlicense, CC0, ISC license, and Artistic License 2.0."


* Llama

   Github:
      Disk size - 328 GB
      Epochs - 0.64
      Sampling proportion - 4.5%
   
   Sampling strategy: "We use the public GitHub dataset available on Google BigQuery. We only kept projects that are distributed under the Apache, BSD and MIT licenses. Additionally, we filtered low quality files with heuristics based on the line length or proportion of alphanumeric characters, and removed boilerplate, such as headers, with regular expressions. Finally, we deduplicate the resulting dataset at the file level, with exact matches."

* PaLM

   Github:
      Sampling proportion - 5%

   Sampling strategy: "The source code in the pretraining dataset is obtained from open source repositories on GitHub. We filtered the files by the license included in the repository; copyleft licenses were excluded. We filter the files by filename extension to restrict to one of 24 common programming languages, including Java, HTML, Javascript, Python, PHP, C#, XML, C++, and C, which results in 196GB of source code. Further, we remove duplicates based on Levenshtein distance between the files because duplicate files are known to be common in source code repositories (Lopes et al., 2017; Allamanis, 2019)."


## References

1. O. Tange (2018): GNU Parallel 2018, March 2018, https://doi.org/10.5281/zenodo.1146014.
2. Rae, Jack W. et al. “Scaling Language Models: Methods, Analysis & Insights from Training Gopher.” ArXiv abs/2112.11446 (2021): n. pag.
3. Touvron, Hugo et al. “LLaMA: Open and Efficient Foundation Language Models.” ArXiv abs/2302.13971 (2023): n. pag.
4. Chowdhery, Aakanksha et al. “PaLM: Scaling Language Modeling with Pathways.” ArXiv abs/2204.02311 (2022): n. pag.
