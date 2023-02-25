from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
import springs as sp
from smashed.utils import io_utils
import pandas as pd
import cld3
import json


LANG_ID_CUT = 2000
COMMON_CUT = 100


@sp.dataclass
class ProcessTextConfig:
    src: str = sp.field(default=sp.MISSING, help="Path to S3 prefix containing parqet files")
    dst: str = sp.field(default=sp.MISSING, help="Path to S3 prefix to write parqet files")
    debug: bool = sp.field(default=False, help="Debug mode")
    cpu_count: int = sp.field(default=cpu_count(), help="Number of processes to use")


def process_single(src: str, dst: str):
    df = pd.read_parquet(src)

    # # for debugging purposes, only take first 1000 rows
    df = df.head(100)

    # strip leading and trailing whitespace
    df['text'] = df['text'].str.strip()

    # create new column that is the result of the function
    # cld3.get_language(text) applied to the text column
    df['lang'] = df['text'].apply(
        # only use first LANG_ID_CUT characters to determine language
        lambda x: cld3.get_language(x[:LANG_ID_CUT]).language   # type: ignore
    )

    # whitespace tokenize the text column
    df['tokens'] = df['text'].str.split()

    # get the number of tokens as a new column
    df['cnt'] = df['tokens'].apply(len)

    # get a frequency distribution of the tokens
    df['freq'] = df['tokens'].apply(
        lambda x: json.dumps(
            # gotta store as a json string because parquet doesn't support
            # dicts as is.
            Counter(x).most_common(COMMON_CUT)
        )
    )

    # drop the tokens column
    df = df.drop(columns=['tokens'])

    # write the dataframe to the destination
    df.to_parquet(dst)


@sp.cli(ProcessTextConfig)
def main(cfg: ProcessTextConfig):
    src = io_utils.MultiPath.parse(cfg.src)
    dst = io_utils.MultiPath.parse(cfg.dst)

    src_paths = [p for p in io_utils.recursively_list_files(src)]
    dst_paths = [dst / (single_src - src) for single_src in src_paths]

    if cfg.debug:
        with tqdm(total=len(src_paths)) as pbar:
            for single_src, single_dst in zip(src_paths, dst_paths):
                process_single(single_src.as_str, single_dst.as_str)
                pbar.update(1)

    else:
        # use a process pool to process the files in parallel
        # and keep track of progress with tqdm
        with ProcessPoolExecutor(max_workers=cfg.cpu_count) as executor:
            with tqdm(total=len(src_paths)) as pbar:
                for _ in executor.map(
                    process_single,
                    src_paths,
                    dst_paths
                ):
                    pbar.update(1)


if __name__ == '__main__':
    main()
