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
from dataloader.testdataset import TestDataset
import math
from torch.utils.data import DataLoader
from torchvision import transforms
from models import VAR_RoPE, VQVAE, build_var
from einops import rearrange
import json
from utils.misc import metrics, save_img, gaussian_weights, pt_to_numpy, numpy_to_pil

logger = get_logger(__name__, log_level="INFO")

def main(args: arg_util.Args):
    device = args.device
    vae_ckpt =  args.vae_model_path
    var_ckpt = args.var_test_path
    exp_name = os.path.basename(os.path.dirname(var_ckpt))

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

    img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])

    with open(args.json_path, "r") as f:
        data_dict = json.load(f)["val"]

    items = list(data_dict.items())
    eval_img_num = len(items) if args.eval_img_num is None else min(args.eval_img_num, len(items))
    acc_mean = 0.0
    rscale = 1
    for i in range(len(items[:eval_img_num])):
        key, entry = items[i]
        image_name = entry["render"]
        ref_image_name = entry["ref"]
        gt_image_name = entry["gt"]
        lr_inp = Image.open(image_name).convert("RGB")
        ref_inp = Image.open(ref_image_name).convert("RGB")
        gt = Image.open(gt_image_name).convert("RGB")

        # save render, ref and gt
        save_img(os.path.join(args.testset_path, "renders"), lr_inp, key)
        save_img(os.path.join(args.testset_path, "ref"), ref_inp, key)
        save_img(os.path.join(args.testset_path, "gt"), gt, key)

        ori_h = lr_inp.size[0]*rscale
        ori_w = lr_inp.size[1]*rscale

        lr_inp = lr_inp.resize((max(math.ceil(lr_inp.size[0]/16)*16*rscale,512), max(math.ceil(lr_inp.size[1]/16)*16*rscale, 512))) 
        lr_inp = img_preproc(lr_inp).unsqueeze(0) * 2.0 - 1.0
        lr_inp = lr_inp.to(dist.get_device(), non_blocking=True)

        ref_inp = ref_inp.resize((max(math.ceil(ref_inp.size[0]/16)*16*rscale,512), max(math.ceil(ref_inp.size[1]/16)*16*rscale, 512)))
        ref_inp = img_preproc(ref_inp).unsqueeze(0) * 2.0 - 1.0
        ref_inp = ref_inp.to(dist.get_device(), non_blocking=True)

        label_render = torch.zeros(1).to(dist.get_device(),  dtype = int, non_blocking=True)
        label_ref = torch.ones(1).to(dist.get_device(),  dtype = int, non_blocking=True)
        label_B = torch.cat([label_render, label_ref], dim=0) if args.use_ref else label_render
        B = lr_inp.shape[0]
        h, w = math.ceil(lr_inp.shape[2]/16), math.ceil(lr_inp.shape[3]/16)
        tile_size = 32
        tile_overlap = 8
        tile_weights = gaussian_weights(32, 32, 1)

        grid_rows = 0
        cur_x = 0
        while cur_x < h:
            cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
            grid_rows += 1

        grid_cols = 0
        cur_y = 0
        while cur_y < w:
            cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
            grid_cols += 1
        recon_pred = []
        start_time = time.time()
        for row in range(grid_rows):
            input_lr = []
            input_ref = []
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = h - tile_size
                if col == grid_cols-1:
                    ofs_y = w - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size
                # input tile dimensions
                input_lr_tile = lr_inp[:, :, input_start_x*16:input_end_x*16, input_start_y*16:input_end_y*16]
                input_ref_tile = ref_inp[:, :, input_start_x*16:input_end_x*16, input_start_y*16:input_end_y*16]

                inp = torch.cat([input_lr_tile, input_ref_tile], dim=0) if args.use_ref else input_lr_tile # [(b v), c, h, w]

                print(inp.shape)
                with torch.inference_mode():
                    V = 2 if args.use_ref else 1
                    with torch.autocast('cuda', enabled=True, dtype=torch.float32, cache_enabled=True):    # using bfloat16 can be faster
                        output = var.autoregressive_infer_cfg(B=V, cfg=1.0, top_k=1, top_p=0.96,
                                                            text_hidden=None, lr_inp=inp, negative_text=None, label_B=label_B, lr_inp_scale = None, tile_flag=True,
                                                            more_smooth=False)
                        output = rearrange(output, '(b v) c h w -> b v c h w', v=V)
                        # rm ref img
                        recon_B3HW = output[:, 0]
                        recon_pred.append(recon_B3HW)

        preds = torch.zeros((B, 32, h, w), device=lr_inp.device)
        contributors = torch.zeros((B, 32, h, w), device=lr_inp.device)
        # Add each tile contribution to overall latents
        for row in range(grid_rows):
            for col in range(grid_cols):
                if col < grid_cols-1 or row < grid_rows-1:
                    # extract tile from input image
                    ofs_x = max(row * tile_size-tile_overlap * row, 0)
                    ofs_y = max(col * tile_size-tile_overlap * col, 0)
                    # input tile area on total image
                if row == grid_rows-1:
                    ofs_x = h - tile_size
                if col == grid_cols-1:
                    ofs_y = w - tile_size

                input_start_x = ofs_x
                input_end_x = ofs_x + tile_size
                input_start_y = ofs_y
                input_end_y = ofs_y + tile_size

                preds[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += recon_pred[row * grid_cols + col] * tile_weights #[b, c, h, w]
                contributors[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += tile_weights
        # Average overlapping areas with more than 1 contributor
        preds /= contributors
        with torch.no_grad():
            cond_features = [vae.get_cond_features(lr_inp)] if args.use_bridge else None
            recon_B3HW = vae.fhat_to_img(preds, cond_features).add_(1).mul_(0.5)
        recon_B3HW = numpy_to_pil(pt_to_numpy(recon_B3HW))
        end_time = time.time()
        duration = end_time - start_time
        print(duration)
        for idx in range(B):
            image = recon_B3HW[idx].resize((ori_h, ori_w))
            if True:
                validation_image = Image.open(ref_image_name).convert("RGB")
                validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))
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
