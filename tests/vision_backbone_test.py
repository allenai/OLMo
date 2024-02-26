import pytest
import torch

from olmo.config import ModelConfig, VisionBackboneConfig, ResamplerConfig, ProjectorConfig, PaddingDirection
from olmo.model import OlmoVisionBackbone, BufferCache
from olmo.data.collator import DataCollator
from olmo.mm_data.image_token_size import AnyResImageTokenizer


def test_vision_backbones():
    collator = DataCollator(pad_direction=PaddingDirection.right, pad_token_id=1)
    image_size = 336  # width and height
    patch_size = 14
    
    # Fixed number of tokens
    n_tokens = (image_size // patch_size) * (image_size // patch_size)

    inputs = [
        {
            "input_ids": torch.tensor([1, 2, 3] + [0] * n_tokens + [4]),
            "image_offsets": torch.tensor(range(3, 3+n_tokens)),
            "image_patches": torch.rand(1, 3, image_size, image_size),
            "num_patches_per_image": torch.tensor([1]),
        },
        {
            "input_ids": torch.tensor([4] + [0] * n_tokens + [1, 2] + [0] * n_tokens + [3]),
            "image_offsets": torch.tensor(list(range(1, 1+n_tokens)) + list(range(3+n_tokens, 3+2*n_tokens))),
            "image_patches": torch.rand(2, 3, image_size, image_size),
            "num_patches_per_image": torch.tensor([1, 1]),
        },
    ]
    batch = collator(inputs)  # type: ignore
    v_cfg = VisionBackboneConfig(
        name="ViT-L-14-336",
        pretrained="openai",
        image_width=336,
        image_height=336,
        patch_width=14,
        patch_height=14,
        select_layer=-2,
        anyres=False,
        possible_resolutions=None,
        pad_image=False,
        frozen=True,
    )
    resampler_cfg = ResamplerConfig(
        d_query=1024,
        n_queries=144,
        n_heads=16,
    )
    projector_cfg = ProjectorConfig(
        d_visual=1024,
        n_layers=2,
        activation_type="gelu",
    )
    # config = ModelConfig(vision_backbone=v_cfg, projector=projector_cfg)

    # vision_backbone = OlmoVisionBackbone.build(config, BufferCache())
    # img_emb = vision_backbone(batch["image_patches"], batch["num_patches_per_image"])
    # assert img_emb.shape == (3 * n_tokens, config.d_model)

    # Images of any resolution
    def _get_batch(sz, image_sizes):
        n_tokens = [sz(*s) for s in image_sizes]
        inputs = [
            {
                "input_ids": torch.tensor([1, 2, 3] + [0] * n_tokens[0] + [4]),
                "image_offsets": torch.tensor(range(3, 3+n_tokens[0])),
                "image_patches": torch.rand(2, 3, image_size, image_size),
                "num_patches_per_image": torch.tensor([2]),
                "image_sizes": torch.tensor(image_sizes[0:1]),
            },
            {
                "input_ids": torch.tensor([4] + [0] * n_tokens[1] + [1, 2] + [0] * n_tokens[2] + [3]),
                "image_offsets": torch.tensor(
                    list(range(1, 1+n_tokens[1])) + list(range(3+n_tokens[1], 3+sum(n_tokens[1:])))),
                "image_patches": torch.rand(8, 3, image_size, image_size),
                "num_patches_per_image": torch.tensor([3, 5]),
                "image_sizes": torch.tensor(image_sizes[1:]),
            },
        ]
        batch = collator(inputs)  # type: ignore
        return batch

    possible_resolutions = [
        (image_size*1, image_size*1),
        (image_size*1, image_size*2),
        (image_size*2, image_size*1),
        (image_size*2, image_size*2)
    ]
    sz = AnyResImageTokenizer(image_size, image_size, patch_size, patch_size, possible_resolutions)
    image_sizes = [(336, 336), (336, 672), (448, 448)]
    n_tokens = [sz(*s) for s in image_sizes]

    batch = _get_batch(sz, image_sizes)

    v_cfg.anyres = True
    v_cfg.possible_resolutions = possible_resolutions
    # config = ModelConfig(vision_backbone=v_cfg, projector=projector_cfg)

    # vision_backbone = OlmoVisionBackbone.build(config, BufferCache())
    # img_emb = vision_backbone(batch["image_patches"], batch["num_patches_per_image"], batch["image_sizes"])
    # assert img_emb.shape == (sum(n_tokens), config.d_model)

    sz = AnyResImageTokenizer(image_size, image_size, patch_size, patch_size, possible_resolutions, resampler_cfg.n_queries)
    n_tokens = [sz(*s) for s in image_sizes]
    batch = _get_batch(sz, image_sizes)

    config = ModelConfig(vision_backbone=v_cfg, resampler=resampler_cfg, projector=projector_cfg)
    
    vision_backbone = OlmoVisionBackbone.build(config, BufferCache())
    img_emb = vision_backbone(batch["image_patches"], batch["num_patches_per_image"], batch["image_sizes"])
    assert img_emb.shape == (sum(n_tokens), config.d_model)


