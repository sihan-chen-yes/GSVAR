import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
from torch import nn
from torchvision import transforms
from torch.utils import data as data
import torch.nn.functional as F
from PIL import ImageFile
from transformers import AutoProcessor
ImageFile.LOAD_TRUNCATED_IMAGES = True
import json

def exists(x):
    return x is not None

class TestDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets_share_1/quyunpeng/trainset", 
                image_size=512,
                tokenizer=None,
                null_text_ratio=0.0,
                original_image_ratio = 0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(TestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.original_image_ratio = original_image_ratio
        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.img_paths = []
        self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/HR/*'))[:])

        print(len(self.img_paths))


    def tokenize_caption(self, caption):            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        GT_image = Image.open(img_path).convert('RGB')
        scale = 1.0
        resolution = round(512 * scale)
        GT_image_t = self.img_preproc(GT_image.resize((resolution, resolution)))
        example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0
        example['path'] = img_path

        img_path = self.img_paths[index].replace("/HR/", "/LR/")
        LR_image_t = Image.open(img_path).convert('RGB')
        if LR_image_t.size[-1] != resolution:
            example["conditioning_pixel_values"] = self.img_preproc(LR_image_t.resize((resolution, resolution))).squeeze(0) * 2.0 - 1.0
        else:
            example["conditioning_pixel_values"] = self.img_preproc(LR_image_t).squeeze(0) * 2.0 - 1.0

        example["label_B"] = 0
        return example

    def __len__(self):
        return len(self.img_paths)

class GSTestDataset(data.Dataset):
    def __init__(self,
                json_path="data/data.json",
                image_size=512,
                tokenizer=None,
                null_text_ratio=0.0,
                original_image_ratio = 0.0,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
                args=None,
        ):
        super(GSTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.original_image_ratio = original_image_ratio
        self.args = args
        self.img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

        with open(json_path, "r") as f:
            self.data_dict = json.load(f)["val"]

        self.items = list(self.data_dict.items())

        print(f"img number:{len(self.items)}")


    def tokenize_caption(self, caption):
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()
        scale = 1.0
        resolution = round(512 * scale)

        # load image
        key, entry = self.items[index]

        gt_path = entry["gt"]
        render_path = entry["render"]
        gt_image = Image.open(gt_path).convert('RGB')
        render_image = Image.open(render_path).convert('RGB')
        example['gt_path'] = gt_path
        example['render_path'] = render_path
        example["pixel_values"] = self.img_preproc(gt_image.resize((resolution, resolution))) * 2.0 - 1.0
        example["conditioning_pixel_values"] = self.img_preproc(render_image.resize((resolution, resolution))) * 2.0 - 1.0
        example['key'] = key

        example["label_B"] = torch.tensor(0, dtype=torch.long)
        if self.args.use_ref:
            if self.args.use_ref_drop and random.random() < 0.5:
                example['ref_path'] = render_path
                example["pixel_values"] = torch.stack([example["pixel_values"], example["pixel_values"]], dim=0)
                example["conditioning_pixel_values"] = torch.stack([example["conditioning_pixel_values"], example["conditioning_pixel_values"]], dim=0)
                example["label_B"] = torch.stack([example["label_B"], torch.tensor(0, dtype=torch.long)], dim=0)
            else:
                ref_path = entry.get("ref", None)

                example['ref_path'] = ref_path
                ref_image = Image.open(ref_path).convert('RGB')
                ref_image = self.img_preproc(ref_image.resize((resolution, resolution))) * 2.0 - 1.0
                example["pixel_values"] = torch.stack([example["pixel_values"], ref_image], dim=0)
                example["conditioning_pixel_values"] = torch.stack([example["conditioning_pixel_values"], ref_image], dim=0)
                example["label_B"] = torch.stack([example["label_B"], torch.tensor(1, dtype=torch.long)], dim=0)
        else:
            example["pixel_values"] = torch.stack([example["pixel_values"]], dim=0)
            example["conditioning_pixel_values"] = torch.stack([example["conditioning_pixel_values"]], dim=0)
            example["label_B"] = torch.stack([example["label_B"]], dim=0)

        return example

    def __len__(self):
        return len(self.items)
