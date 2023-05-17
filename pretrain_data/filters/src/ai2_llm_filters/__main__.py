import argparse

from .core_tools.registry import TaggerRegistry
from .core_tools.runtime import TaggerProcessor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        required=True,
        help=f"Name dataset to process; this should be relative path from {TaggerProcessor.BASE_S3_PREFIX}.",
    )
    ap.add_argument(
        "-t",
        "--tagger",
        required=True,
        nargs="+",
        help="One or more taggers to run; use -l to list available taggers.",
    )
    ap.add_argument("-l", "--list-taggers", action="store_true", help="List available taggers.")
    ap.add_argument("-p", "--parallel", type=int, default=1, help="Number of parallel processes to use.")
    ap.add_argument("-d", "--debug", action="store_true", help="Run in debug mode; parallelism will be disabled.")
    opts = ap.parse_args()

    if opts.list_taggers:
        print("Available taggers:")
        for tagger_name, tagger_cls in TaggerRegistry.taggers():
            print(f"  {tagger_name} ({tagger_cls.__name__})")
        return

    TaggerProcessor.main(
        dataset=opts.dataset,
        taggers=opts.tagger,
        num_processes=opts.parallel,
        debug=opts.debug,
    )


if __name__ == "__main__":
    main()
