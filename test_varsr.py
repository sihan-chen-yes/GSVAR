import os
import sys
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
import torch.nn.functional as F
import safetensors.torch
import time
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
import dist
import torch
from torchvision import transforms
import torch.utils.checkpoint
from utils import arg_util, misc
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import CLIPTextModel, CLIPVisionModel, CLIPTokenizer, CLIPImageProcessor
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from dataloader.testdataset import TestDataset, GSTestDataset
import math
from torch.utils.data import DataLoader
from torchvision import transforms
import pyiqa
from skimage import io
from models import VAR_RoPE, VQVAE, build_var
from einops import rearrange, repeat
from torch.utils.data import Subset
from utils.misc import metrics, save_img, pt_to_numpy, numpy_to_pil
from torchvision.transforms.functional import to_pil_image, to_tensor
logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args):
    vae_ckpt =  args.vae_model_path
    var_ckpt = args.var_test_path
    exp_name = f'{os.path.basename(os.path.dirname(var_ckpt))}_org'

    vae, var = build_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth=args.depth,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums, control_patch_nums =args.patch_nums,
        num_classes=1 + 1, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
        args=args,
    )
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local'], strict=True)
    model_state = torch.load(var_ckpt, map_location='cpu')
    print(f"loading var ckpt at epoch:{model_state['epoch']}, iter:{model_state['iter']}...")
    var.load_state_dict(model_state['trainer']['var_wo_ddp'], strict=True)
    vae.eval(), var.eval()

    eval_img_num = args.eval_img_num
    dataset_val = GSTestDataset(json_path=args.json_path, image_size=args.data_load_reso, tokenizer=None, resize_bak=True, args=args) # for gs enhancement
    dataset_len = len(dataset_val) if eval_img_num is None else min(eval_img_num, len(dataset_val))
    dataset_val = Subset(dataset_val, list(range(dataset_len)))
    ld_val = DataLoader(
        dataset_val, num_workers=8, pin_memory=True,
        batch_size=round(args.batch_size), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )

    for batch in ld_val:
        lr_inp = batch["conditioning_pixel_values"].to(args.device, non_blocking=True)
        label_B = batch["label_B"].to(args.device, non_blocking=True)
        B, V, C, H, W = lr_inp.shape
        lr_inp = rearrange(lr_inp, 'b v c h w -> (b v) c h w')
        label_B = rearrange(label_B, 'b v-> (b v)')

        start_time = time.time()
        with torch.no_grad():
            recon_B3HW = var.autoregressive_infer_cfg(B=B * V, cfg=1.0, top_k=1, top_p=0.96,
                                                text_hidden=None, lr_inp=lr_inp, negative_text=None, label_B=label_B, lr_inp_scale = None,
                                                more_smooth=False)
            recon_B3HW = rearrange(recon_B3HW, '(b v) c h w -> b v c h w', v=V)[:, 0]
            recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))

        end_time = time.time()
        duration = end_time - start_time
        print(duration)
        for idx in range(B):
            render_img = Image.open(batch['render_path'][idx]).convert("RGB")
            ref_img = Image.open(batch['ref_path'][idx]).convert("RGB")
            gt_img = Image.open(batch['gt_path'][idx]).convert("RGB")
            key = batch['key'][idx]
            # save render, ref and gt
            save_img(os.path.join(args.testset_path, "renders"), render_img, key)
            save_img(os.path.join(args.testset_path, "ref"), ref_img, key)
            save_img(os.path.join(args.testset_path, "gt"), gt_img, key)

            image = recon_B3HW[idx]

            # asset export
            # gt = batch["pixel_values"].to(args.device, non_blocking=True)
            # gt = rearrange(gt, 'b v c h w -> (b v) c h w')
            # _, gt_idx = vae.img_to_idxBl(gt, args.patch_nums)
            # gt_level_img = vae.idxBl_list_to_img(gt_idx, same_shape=True, last_one=False)
            # for i, level_img in enumerate(gt_level_img):
            #     save_img(os.path.join("dbg", "asset"), to_pil_image(level_img[0].add_(1).mul_(0.5)), f'gt_{i}')
            # _, pred_img_idx = vae.img_to_idxBl(to_tensor(image).mul_(2).sub_(1).to(args.device, non_blocking=True)[None, ...], args.patch_nums)
            # pred_level_img = vae.idxBl_list_to_img(pred_img_idx, same_shape=True, last_one=False)
            # for i, level_img in enumerate(pred_level_img):
            #     save_img(os.path.join("dbg", "asset"), to_pil_image(level_img[0].add_(1).mul_(0.5)), f'pred_{i}')

            if True:
                validation_image = Image.open(batch['ref_path'][idx]).convert("RGB")
                w, h = validation_image.size
                image = image.resize((w, h), Image.LANCZOS)
                image = adain_color_fix(image, validation_image)

            output_dir = os.path.join(args.testset_path, "VARPrediction", exp_name, f"{model_state['epoch']}_{model_state['iter']}")
            save_img(output_dir, image, key)

    info = {
        "epoch": model_state['epoch'],
        "iter": model_state['iter'],
        "exp_name": exp_name,
    }

    return info

if __name__ == "__main__":
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.gen_img:
        info = main(args)
    else:
        info = {
            "epoch": args.epoch,
            "iter": args.iter,
            "exp_name": args.exp_name,
        }
    results = metrics(args, info)
