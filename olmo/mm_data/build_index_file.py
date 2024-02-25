import argparse
import dataclasses
import json
import logging
from os import listdir
from os.path import exists, join
from time import perf_counter

from olmo.mm_data.data_store import ExampleReader, MMStorageConfig
from olmo.mm_data.image_token_size import FixedNumberOfToken
from olmo.mm_data.sequence_index import get_idx_file, build_sequence_index, MMDatasetConfig
from olmo.mm_data.structure_index import BasicIndexer
from olmo.util import prepare_cli_environment


logger = logging.getLogger(__name__)


def get_sizer(size_str):
    if size_str.startswith("fixed"):
        return FixedNumberOfToken(int(size_str[len("fixed"):]))
    else:
        raise NotImplementedError(size_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafiles", nargs="+")
    parser.add_argument("-s", "--seed", type=int, nargs="+")
    parser.add_argument("-l", "--seq_len", type=int)
    parser.add_argument("-sz")
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    prepare_cli_environment()

    idx_dir = args.output_dir

    data = MMDatasetConfig(args.datafiles)
    param_file = join(idx_dir, MMDatasetConfig.FILENAME)
    if exists(param_file):
        with open(param_file) as f:
            existing = json.load(f)
        assert existing == data.as_dict()
    else:
        if any(x.startswith("index.") for x in listdir(idx_dir)):
            logger.warning(f"There appear to be index files in {idx_dir}, but no data config, The data config will "
                           "be re-written, make sure all these index files are for the same data config")
        with open(param_file, "w") as f:
            json.dump(data.as_dict(), f)

    paths = args.datafiles
    seed = args.seed
    image_sizer = get_sizer(args.sz)

    for seed in args.seed:
        output_file = join(args.output_dir, get_idx_file(image_sizer, args.seq_len, seed))
        if exists(output_file):
            logger.info(f"Already have index file for seed {seed}: {output_file}")
            return
        t0 = perf_counter()
        logger.info(f"Starting shuffle {seed}")
        build_sequence_index(
            data,
            args.seq_len,
            args.seed,
            image_sizer,
            BasicIndexer(),
            output_file
        )
        logger.info(f"Done in {perf_counter() - t0:0.2f} seconds")


if __name__ == '__main__':
    main()
