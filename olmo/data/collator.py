from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F

from ..config import PaddingDirection, TrainConfig

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    pad_direction: PaddingDirection
    pad_token_id: int

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> DataCollator:
        return cls(pad_direction=config.data.pad_direction, pad_token_id=config.model.pad_token_id)

    def __call__(self, items: Union[List[Dict[str, Any]], List[torch.Tensor]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        max_images = 0
        max_tokens = 0
        max_patches = 0
        if items and isinstance(items[0], dict) and "num_patches_per_image" in items[0]:
            max_images = max(len(x["num_patches_per_image"]) for x in items)  # type: ignore
            max_tokens = max(len(x["image_offsets"]) for x in items) # type: ignore
            max_patches = max(len(x["image_patches"]) for x in items) # type: ignore
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        all_label_mask = []
        all_image_patches = []
        all_image_offsets = []
        all_image_sizes = []
        all_num_patches_per_image = []
        all_indices = []
        all_metadata = []
        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float),
                        pad_shape,
                        value=0.0,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

            # Pad label mask.
            label_mask = x.get("label_mask") if isinstance(x, dict) else None
            if label_mask is not None:
                if not isinstance(label_mask, torch.Tensor):
                    label_mask = torch.tensor(label_mask)
                all_label_mask.append(
                    F.pad(
                        label_mask.to(dtype=torch.bool),
                        pad_shape,
                        value=False,
                    )
                )

            # Image patches, offsets, sizes, num_patches_per_image
            num_patches_per_image = x.get("num_patches_per_image") if isinstance(x, dict) else None
            image_sizes = x.get("image_sizes") if isinstance(x, dict) else None
            if num_patches_per_image is not None:
                num_patches_per_image = F.pad(
                    num_patches_per_image.to(dtype=torch.int32),
                    (0, max_images - len(num_patches_per_image)),
                    value=0,
                )
                all_num_patches_per_image.append(num_patches_per_image)
                image_offsets = F.pad(
                    x["image_offsets"].to(dtype=torch.int32),
                    (0, max_tokens - len(x["image_offsets"])),
                    value=-1,
                )
                all_image_offsets.append(image_offsets)
                image_patches = F.pad(
                    x["image_patches"].to(dtype=torch.float),
                    (0, 0, 0, 0, 0, 0) + (0, max_patches - len(x["image_patches"])),
                    value=0.0,
                )
                all_image_patches.append(image_patches)
                if image_sizes is not None:
                    image_sizes = F.pad(
                        image_sizes.to(dtype=torch.int32),
                        (0, 0) + (0, max_images - len(image_sizes)),
                        value=0,
                    )
                    all_image_sizes.append(image_sizes)

            # Indices.
            index = x.get("index") if isinstance(x, dict) else None
            if index is not None:
                all_indices.append(torch.tensor(index))

            # Metadata.
            metadata = x.get("metadata") if isinstance(x, dict) else None
            if metadata is not None:
                all_metadata.append(metadata)

        """
        Out: Dict[str, Any]
        - input_ids: (batch_size, sequence_length,)
        - label_mask: (batch_size, sequence_length,)
        - image_patches: (batch_size, num_patches, 3, height, width)
            num_patches is the (maximum) number of patches in each sequence
        - image_offsets: (batch_size, n_tokens)
            n_tokens is the (maximum) number of image tokens in each sequence
        - num_patches_per_image: (batch_size, n_images,)
            n_images is the (maximum) number of images in each sequence
        - image_sizes: (batch_size, num_images, 2)
            n_images is the (maximum) number of images in each sequence
            width, height of each image
        """
        out: Dict[str, Any] = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        if all_label_mask:
            out["label_mask"] = torch.stack(all_label_mask)
        if all_image_patches:
            out["image_patches"] = torch.stack(all_image_patches)
        if all_image_offsets:
            out["image_offsets"] = torch.stack(all_image_offsets)
        if all_num_patches_per_image:
            out["num_patches_per_image"] = torch.stack(all_num_patches_per_image)
        if all_image_sizes:
            out["image_sizes"] = torch.stack(all_image_sizes)
        if all_indices:
            out["index"] = torch.stack(all_indices)
        if all_metadata:
            out["metadata"] = all_metadata

        return out
