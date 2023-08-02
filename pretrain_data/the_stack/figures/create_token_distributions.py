import argparse
import pandas as pd
import seaborn as sns


def _create_lang_size_dict(file_sizes_path: str) -> dict:
    with open(file_sizes_path) as f:
        lines = [l.strip() for l in f.readlines()]

    path_dict = {}
    for i, l in enumerate(lines[:-4]):
        parts = l.split(" ")
        size_n = parts[-3]
        size_u = parts[-2]
        lang_file = "/".join(parts[-1].split("/")[-2:])
        path_dict[lang_file] = (size_n, size_u)

    multiplier = {
        "GiB": 1,
        "MiB": 1/1024,
        "KiB": 1/(1024*1024),
        "Bytes": 1/(1024*1024*1024)
    }

    path_size_dict = {}
    for key, val in path_dict.items():
        size = float(val[0]) * multiplier[val[1]]
        path_size_dict[key] = size

    lang_size_dict = {}
    for key, val in path_size_dict.items():
        lang  = key.split("/")[0]
        if lang in lang_size_dict:
            lang_size_dict[lang] += val
        else:
            lang_size_dict[lang] = val

    return lang_size_dict


def create_figure(file_sizes_path: str, version: str, output_file: str, top_n: int = 30):
    
    lang_size_dict = _create_lang_size_dict(file_sizes_path)
    sorted_lang_sizes = sorted(lang_size_dict.items(), key=lambda x:x[1], reverse=True)

    top_n_langs = [tup[0] for tup in sorted_lang_sizes[:top_n]]

    top_n_dict = {key: lang_size_dict[key] for key in top_n_langs}
    df = pd.DataFrame.from_dict(top_n_dict, orient="index", columns=["size"])
    df = df.reset_index().rename(columns={"index": "lang"})

    df["tokens"] = df["size"] / 2  # 2 GB == 1 B tokens

    ax = sns.barplot(df, x="lang", y="tokens", estimator=sum, errorbar=None, color="lightblue")
    ax.set(xlabel="Language", ylabel="Number of tokens (billions)")
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f"Distribution of top 30 programming languages in stack-{version}")
    ax.figure.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Create token distribution figure")
    parser.add_argument("input_file", type=str, help="Output of aws s3 ls <tokenized-data-s3-path> --recursive --human-readable --summarize > file.txt")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--top-n", type=int, required=False, default=30)

    args = parser.parse_args()
    create_figure(args.input_file, args.version, args.output_file, args.top_n)
