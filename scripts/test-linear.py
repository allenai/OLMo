import math
import os
from typing import List
import torch
import torch.nn.functional as F


def copy_large_layer_weights_to_small(
        large_layer: torch.nn.Module,
        small_layers: List[torch.nn.Module]) -> None:

    with torch.no_grad():
        large_layer_chunks = large_layer.weight.chunk(len(small_layers))

        for i, small_layer in enumerate(small_layers):
            small_layer.weight.copy_(large_layer_chunks[i])


def copy_small_layer_weights_to_large(
        large_layer: torch.nn.Module,
        small_layers: List[torch.nn.Module]) -> None:

    with torch.no_grad():
        combined_small_layer_weights = torch.cat(
            [
                small_layer.weight
                for small_layer in small_layers
            ],
            dim=0,
        )
        large_layer.weight.copy_(combined_small_layer_weights)


def main():
    torch.manual_seed(42)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # needed for running in the deterministic mode
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True

    dtype = torch.bfloat16
    # dtype = torch.float32
    device = torch.device('cuda')

    batch_size = 65703
    input_dim = 6747
    small_layer_output_dim = 427
    num_small_layers = 4

    large_layer_output_dim = small_layer_output_dim * num_small_layers
    large_layer = torch.nn.Linear(input_dim, large_layer_output_dim, bias=False, device=device)
    torch.nn.init.trunc_normal_(large_layer.weight, std=0.02)

    small_layers = [
        torch.nn.Linear(input_dim, small_layer_output_dim, bias=False, device=device)
        for _ in range(num_small_layers)
    ]

    torch.nn.init.trunc_normal_(large_layer.weight, std=0.02)
    copy_large_layer_weights_to_small(large_layer, small_layers)

    # for small_layer in small_layers:
    #     torch.nn.init.trunc_normal_(small_layer.weight, std=0.02)
    # copy_small_layer_weights_to_large(large_layer, small_layers)

    input_vector = torch.randn(batch_size, input_dim, device=device)

    # with torch.autocast(device.type, dtype=dtype):
    #     large_layer_output = large_layer(input_vector)
    #     small_layers_output = torch.cat([
    #         small_layer(input_vector)
    #         for small_layer in small_layers
    #     ], dim=-1)

    # assert torch.allclose(large_layer_output, small_layers_output), (large_layer_output, small_layers_output)

    # print(torch.max(torch.abs(large_layer_output - small_layers_output)))

    with torch.autocast(device.type, dtype=dtype):
        large_layer_output_split = large_layer(input_vector).split(small_layer_output_dim, dim=-1)
        small_layers_outputs = [
            small_layer(input_vector)
            for small_layer in small_layers
        ]

    for large_output, small_output in zip(large_layer_output_split, small_layers_outputs):
        assert torch.allclose(large_output, small_output), (large_output, small_output)


if __name__ == "__main__":
    main()