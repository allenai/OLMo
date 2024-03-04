import argparse
import logging
from os.path import exists

from olmo.mm_data.data_store import ExampleReader, MMStorageConfig, read_data_file
from olmo.mm_data.structure_index import VectorizedIndexer, get_index_file
from olmo.util import prepare_cli_environment


def main():
    parser = argparse.ArgumentParser("Build or re-build a structure index for a datafile")
    parser.add_argument("data_file")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--override", action="store_true")
    args = parser.parse_args()

    prepare_cli_environment()

    output_file = args.output_file
    if not output_file:
        output_file = get_index_file(args.data_file)
        logging.info(f"Writing to default index location: {output_file}")
    if exists(output_file) and not args.override:
        raise FileExistsError(output_file)

    logging.info("Reading data...")
    # TODO this is not very scalable since it loads the entire dataset as python objects
    data = read_data_file(args.data_file, 0, -1, MMStorageConfig())

    logging.info(f"Writing index for {len(data)} examples...")
    indexer = VectorizedIndexer()
    indexer.write_index(output_file, data)
    logging.info("Done")


if __name__ == '__main__':
    main()