import logging

import torch

from olmo.aliases import PathOrStr
from olmo.safetensors_util import state_dict_to_safetensors_file


def main(input: PathOrStr, output: PathOrStr):
    state_dict = torch.load(input)
    state_dict_to_safetensors_file(state_dict, output)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog=__file__, description="Convert state dicts in pt format to state dicts in safetensors format."
    )
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.input, args.output)
