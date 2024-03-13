from os.path import join
from typing import Dict, Optional
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import transformers

from olmo.config import ModelConfig, DataConfig
from olmo.data import build_image_preprocessor
from olmo.mm_data.image_preprocessing import ImagePreprocessor
from olmo.mm_data.preprocess import Masked, ImageFile


class MMDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer: transformers.PreTrainedTokenizer,
        sequence_length: int,
        image_preprocessor: Optional[ImagePreprocessor] = None,
    ):
        """
        data_path: Path to the dataset file
        tokenizer: Tokenizer to use for text data
        sequence_length: maximum sequence length of examples to yield
        image_preprocessor: How to pre-process images
        """
        super(MMDataset, self).__init__()
        self.sequence_length = sequence_length
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        if image_preprocessor:
            self.image_preprocessor = image_preprocessor
            self.image_sizer = image_preprocessor.image_token_sizer()
        else:
            self.image_preprocessor = None
            self.image_sizer = None

        with open(data_path, 'r') as f:
            self.list_data_dict = json.load(f)

    def __len__(self):
        return len(self.list_data_dict)
    
    def parse_pretrain_data(self, data_dict):
        conv = data_dict['conversations'][1]['value'].strip()
        if self.image_preprocessor:
            example = [
                Masked(self.tokenizer.eos_token),
                ImageFile(join(self.image_dir, data_dict['image'])),
                conv + self.tokenizer.eos_token,
            ]
        else:
            example = [Masked(self.tokenizer.eos_token), conv + self.tokenizer.eos_token]
        return example

    def preprocess(self, item):
        """
        Item: Dict[str, Any]
        - input_ids: (sequence_length,)
        - label_mask: (sequence_length,)
        - image_patches: (num_patches, 3, height, width)
            num_patches is the number of patches from all images in the sequence
        - image_offsets: (n_tokens,)
            n_tokens is the number of image tokens in the sequence
        - num_patches_per_image: (n_images,)
        - image_sizes: (n_images, 2)
            width, height of each image
        """
        if self.image_preprocessor:
            images = item.pop("images")
            offsets = item.pop("image_offsets")
            sizes = item.pop("image_sizes", None)
            if images:
                all_patches = []
                all_patch_offsets = []
                all_num_patches_per_image = []
                for image, offset in zip(images, offsets):
                    patches, patch_offsets = self.image_preprocessor(image, offset)
                    all_patches.append(torch.as_tensor(patches))
                    all_patch_offsets.append(torch.as_tensor(patch_offsets))
                    all_num_patches_per_image.append(patches.shape[0])
                item["image_patches"] = torch.cat(all_patches)
                item["image_offsets"] = torch.cat(all_patch_offsets)
                item["num_patches_per_image"] = torch.as_tensor(all_num_patches_per_image)
                if sizes is not None:
                    item["image_sizes"] = torch.as_tensor(sizes)
        else:
            # text-only mode
            assert len(item["images"]) == 0
            del item["image_offsets"]
            del item["images"]

        # Convert to a torch-compatible dtype
        item["input_ids"] = torch.as_tensor(item["input_ids"].astype(np.int32))
        item["label_mask"] = torch.as_tensor(item["label_mask"])
        return item
    
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        data_dict = self.list_data_dict[i]
        # TODO: Implement parsing for visual instruction tuning data
        example = self.parse_pretrain_data(data_dict)
        indices = np.zeros(self.sequence_length, dtype=np.uint16)
        mask = np.zeros(self.sequence_length, dtype=np.bool_)
        images = []
        sizes = None
        if self.image_sizer is not None and 'anyres' in self.image_sizer.get_id():
            sizes = []
        offsets = []
        
        total_tokens = 0
        prev_was_text = False
        is_masked = None
        for chunk in example:
            if isinstance(chunk, (str, Masked)):
                if isinstance(chunk, str):
                    if prev_was_text and not is_masked:
                        raise ValueError("Consecutive text spans")
                    is_masked = False
                    text = chunk
                else:
                    if prev_was_text and is_masked:
                        raise ValueError("Consecutive masked text spans")
                    is_masked = True
                    text = chunk.text
                if not text:
                    raise ValueError("Got empty text")
                if prev_was_text and not text[0].isspace():
                    raise ValueError("Text must start with a space, or be preceded by an image, "
                                    "or start the document")
                tokens = np.array(self.tokenizer.encode(text, add_special_tokens=False), np.uint16)
                num_tokens = len(tokens)
                token_end = min(total_tokens + num_tokens, self.sequence_length) - total_tokens
                indices[total_tokens:total_tokens + num_tokens] = tokens[:token_end]
                if not is_masked:
                    mask[total_tokens:total_tokens + num_tokens] = True
                prev_was_text = True
            else:
                prev_was_text = False
                offsets.append(total_tokens)
                image = Image.open(chunk.image_file).convert('RGB')
                images.append(image)
                if sizes is not None:
                    sizes.append(np.array(image.size), dtype=np.int32)
                num_tokens = self.image_sizer(image.size[0], image.size[1])
            total_tokens += num_tokens
        
        if not prev_was_text or is_masked:
            raise ValueError("Document must end with a non-masked span")

        total_tokens = min(total_tokens, self.sequence_length)
        item = dict(
            input_ids=indices[:total_tokens],
            label_mask=mask[:total_tokens],
            images=images,
            image_offsets=np.asarray(offsets, np.int32) if offsets else np.zeros((0,), np.int32),
        )
        if sizes is not None:
            item["image_sizes"] = np.stack(sizes) if sizes else np.zeros((0, 2), np.int32)
        
        return self.preprocess(item)


def build_train_dataset(
    tokenizer: transformers.PreTrainedTokenizer,
    model_cfg: ModelConfig,
    data_cfg: DataConfig
) -> Dataset:
    if model_cfg.vision_backbone is not None:
        image_preprocessor = build_image_preprocessor(model_cfg)
    else:
        image_preprocessor = None
    assert len(data_cfg.paths) == 1
    return MMDataset(
        data_path=data_cfg.paths[0],
        image_dir=data_cfg.image_dir,
        tokenizer=tokenizer,
        sequence_length=model_cfg.max_sequence_length,
        image_preprocessor=image_preprocessor,
    )