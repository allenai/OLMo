import json
import os
from random import random
from time import time
from mmda.types import Document
from mmda.recipes import CoreRecipe
import click
import torch
import tqdm

from .io_utils import (
    recursively_list_files, open_file_for_read, open_file_for_write
)


@click.command()
@click.option('--input-prefix', type=str, required=True)
@click.option('--index', type=int, required=True)
@click.option('--output-prefix', type=str, required=True)
@click.option(
    '--device-name', type=str,
    default='cuda' if torch.cuda.is_available() else 'cpu'
)
def main(
    input_prefix: str,
    index: int,
    output_prefix: str,
    device_name: str
):
    # paths = list(recursively_list_files(input_prefix))
    # if index >= len(paths):
    #     raise ValueError(f'index {index} is out of range')

    # with open_file_for_read(paths[index]) as f:
    #     ids = [line.strip() for line in f]

    path_ids = list(recursively_list_files(input_prefix))[305:]

    recipe = CoreRecipe()

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    # move to CUDA
    device = torch.device(device_name)
    recipe.vila_predictor.predictor.device = device
    recipe.vila_predictor.predictor.model.to(device)
    recipe.vila_predictor.predictor.model.eval()
    recipe.effdet_mfd_predictor.model.device = device
    recipe.effdet_mfd_predictor.model.model.to(device)
    recipe.effdet_mfd_predictor.model.model.eval()
    recipe.effdet_publaynet_predictor.model.device = device
    recipe.effdet_publaynet_predictor.model.model.to(device)
    recipe.effdet_publaynet_predictor.model.model.eval()

    data = []
    start = time()
    failures = []
    with torch.inference_mode():
        for doc_id in tqdm.tqdm(path_ids):
            try:
                doc = recipe.from_path(doc_id)
                data.append(doc.to_json())
            except Exception:
                print(doc_id)
                failures.append(doc_id)
    end = time()
    print(f'elapsed time: {end - start:.2f} seconds')
    print(f'failures: {len(failures):,}')
    print(f'correct: {len(data):,}')

    with open(os.path.join(output_prefix, 'failures.txt'), 'w') as f:
        f.write('\n'.join(failures))



if __name__ == '__main__':
    main()
