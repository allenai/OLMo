import argparse
import logging
from os.path import exists, join
from time import perf_counter

from olmo.mm_data.data_iteration import IterationConfig, build_iteration_order, SequenceBuilderConfig, \
    SequenceBuilderKind
from olmo.mm_data.image_token_size import FixedNumberOfToken, AnyResImageTokenizer
from olmo.mm_data.sequence_index import get_idx_file, write_index
from olmo.mm_data.structure_index import VectorizedIndexer
from olmo.util import prepare_cli_environment

logger = logging.getLogger(__name__)


def get_sizer(size_str):
    if size_str.startswith("fixed"):
        return FixedNumberOfToken(int(size_str[len("fixed"):]))
    if size_str == "llava":
        image_size = 336
        return AnyResImageTokenizer(336, 336, 14, 14,  [
            (image_size*1, image_size*1),
            (image_size*1, image_size*2),
            (image_size*2, image_size*1),
            (image_size*2, image_size*2)
        ], 144)
    else:
        raise NotImplementedError(size_str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datafiles", nargs="+")
    parser.add_argument("-s", "--seed", type=int, nargs="+")
    parser.add_argument("-l", "--seq_len", type=int)
    parser.add_argument("-sz")
    parser.add_argument("-o", "--output_dir", required=True)
    parser.add_argument("-n", "--n_processes", default=None, type=int)
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--sequences", choices=list(SequenceBuilderKind))
    parser.add_argument("--pool_size", type=int)
    parser.add_argument("--n_splits", type=int)
    args = parser.parse_args()

    prepare_cli_environment()

    idx_dir = args.output_dir

    data = IterationConfig(
        args.datafiles,
        sequence_builder=SequenceBuilderConfig(
            args.sequences, n_splits=args.n_splits, pool_size=args.pool_size)
    )

    paths = args.datafiles
    seed = args.seed
    image_sizer = get_sizer(args.sz)

    for seed in args.seed:
        output_file = join(args.output_dir, get_idx_file(image_sizer, args.seq_len, seed))
        if exists(output_file) and not args.override:
            logger.info(f"Already have index file for seed {seed}: {output_file}")
            return
        t0 = perf_counter()
        logger.info(f"Starting shuffle {seed}")
        example_arr = build_iteration_order(
            data,
            args.seq_len,
            args.seed,
            image_sizer,
            n_processes=args.n_processes
        )
        logger.info(f"Writing...")
        write_index(example_arr, output_file)
        logger.info(f"Done")


if __name__ == '__main__':
    main()
