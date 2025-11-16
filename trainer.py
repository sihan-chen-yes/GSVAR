import time
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms.functional import crop
import random
import dist
from models import VAR_RoPE, VQVAE, VectorQuantizer2
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModel
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import wandb
import loralib as lora
import torch.nn.functional as F
from test_varsr import metrics, numpy_to_pil, pt_to_numpy
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
from utils.loss import get_gram_loss
from einops import rearrange, repeat
import lpips
Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class BASETrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp,
        text_encoder, clip_vision, exp_name, label_smooth: float, wandb_flag=False,
        fp16: int = 1,
        args = None,
    ):
        super(BASETrainer, self).__init__()

        self.vae_local, self.quantize_local = vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.wandb_flag = wandb_flag
        self.exp_name = exp_name

        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)

        self.text_encoder = text_encoder
        self.clip_vision = clip_vision
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = []
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        self.start_token = self.L - patch_nums[-1] * patch_nums[-1]

        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

        # use ref
        self.args = args
        self.view_num = 2 if self.args.use_ref else 1

        # reconstruction loss
        self.net_lpips = lpips.LPIPS(net='vgg').cuda()
        self.net_lpips.requires_grad_(False)
        self.net_vgg = torchvision.models.vgg16(pretrained=True).features
        for param in self.net_vgg.parameters():
            param.requires_grad_(False)
        self.weight_dtype = torch.float32
        if fp16 == 1:
            self.weight_dtype = torch.float16
        elif fp16 == 2:
            self.weight_dtype = torch.bfloat16
        self.net_lpips = self.net_lpips.to(device, dtype=self.weight_dtype)
        self.net_vgg = self.net_vgg.to(device, dtype=torch.float32)
        self.t_vgg_renorm = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }

    def wandb_report(self, pred_img, lr_inp, gt_inp):
        """
            upload img to wandb
        Args:
            pred_img: output img
            lr_inp: input img
            gt_inp: gt img

        """
        if dist.is_master():
            pred = (rearrange(pred_img, '(b v) c h w -> b v c h w', v=self.view_num)[
                        0].detach().cpu() + 1) / 2.0
            gt = (rearrange(gt_inp, '(b v) c h w -> b v c h w', v=self.view_num)[0].detach().cpu() + 1) / 2.0
            lr = (rearrange(lr_inp, '(b v) c h w -> b v c h w', v=self.view_num)[0].detach().cpu() + 1) / 2.0
            pred = pred.clamp_(0, 1)
            gt = gt.clamp_(0, 1)
            lr = lr.clamp_(0, 1)

            wandb.log({
                "val_render_view/lr": wandb.Image(lr[0]),
                "val_render_view/pred": wandb.Image(pred[0]),
                "val_render_view/gt": wandb.Image(gt[0]),
            })

            if self.view_num == 2:
                wandb.log({
                    "val_ref_view/lr": wandb.Image(lr[1]),
                    "val_ref_view/pred": wandb.Image(pred[1]),
                    "val_ref_view/gt": wandb.Image(gt[1]),
                })

    def get_img_log_dict(self, lr, gt, pred, stage="train"):
        """

        Args:
            range [-1, 1]
            lr: b v c h w
            gt: b v c h w
            pred: b v c h w
            stage: str

        Returns:
            dict for wandb logging
        """
        B = rearrange(lr, "(b v) c h w -> b c (v h) w", v=self.view_num).shape[0]
        log_dict = {
            f"{stage}/input": [
                wandb.Image(
                    (rearrange(lr, "(b v) c h w -> b c (v h) w", v=self.view_num)[
                         idx].float().detach().cpu() + 1.0) / 2,
                    caption=f"idx={idx}") for idx in range(B)],
            f"{stage}/gt": [
                wandb.Image(
                    (rearrange(gt, "(b v) c h w -> b c (v h) w", v=self.view_num)[
                         idx].float().detach().cpu() + 1.0) / 2,
                    caption=f"idx={idx}") for idx in range(B)],
            f"{stage}/output": [
                wandb.Image(
                    (rearrange(pred, "(b v) c h w -> b c (v h) w", v=self.view_num)[
                         idx].float().detach().cpu() + 1.0) / 2,
                    caption=f"idx={idx}") for idx in range(B)],
        }

        return log_dict

class VARTrainer(BASETrainer):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp, var: DDP, 
        text_encoder, clip_vision, exp_name,
        var_opt: AmpOptimizer, label_smooth: float, wandb_flag=False,
        fp16: int = 1,
        args = None,
    ):
        super().__init__(
            device=device, patch_nums=patch_nums, resos=resos,
            vae_local=vae_local, var_wo_ddp=var_wo_ddp,
            text_encoder=text_encoder, clip_vision=clip_vision, exp_name=exp_name, label_smooth=label_smooth,
            wandb_flag=wandb_flag, fp16=fp16, args=args
        )
        self.var = var
        self.var_opt = var_opt

    # teacher-forcing training
    def train_step(
            self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger,
            gt_inp: FTen, lr_inp: Union[ITen, FTen], label_B,
            text, prog_si: int, prog_wp_it: float, lr, wd,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        # forward
        B, V = gt_inp.shape[0], self.vae_local.vocab_size
        self.var.require_backward_grad_sync = stepping
        
        with torch.no_grad():
            gt_idx_Bl, idx_N_list = self.vae_local.img_to_idxBl(gt_inp)
            gt_BL = torch.cat(gt_idx_Bl[0:len(self.patch_nums)], dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(idx_N_list)

        with self.var_opt.amp_ctx:
            self.var_wo_ddp.forward
            ret = self.var(x_BLCv_wo_first_l, label_B, lr_inp, gt_inp, text_hidden=None,
                           last_layer_gt = gt_idx_Bl[-1], last_layer_gt_discrete = gt_idx_Bl[-2], lr_inp_scale = None)

            logits_BLV = ret["logits_BLV"]
            diff_loss = ret["diff_loss"]
            pred = ret["pred_img"]

            if not self.args.use_ref_view_loss:
                # rm ref view img, only eval render view img
                logits_BLV = rearrange(logits_BLV, '(b v) l c -> b v l c', v=self.view_num)[:, 0]
                gt_BL = rearrange(gt_BL, '(b v) l -> b v l', v=self.view_num)[:, 0]
                pred_img = rearrange(pred, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
                gt_img = rearrange(gt_inp, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
            else:
                pred_img = pred
                gt_img = gt_inp

        # Reconstruction loss
        l2_loss = F.mse_loss(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype), reduction="mean")
        lpips_loss = self.net_lpips(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype)).mean()

        with torch.cuda.amp.autocast(enabled=False):
            if self.prog_it > prog_wp_it and self.args.use_gram_loss:
                x_tgt_pred_renorm = self.t_vgg_renorm(pred_img * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
                crop_h, crop_w = 400, 400
                top, left = random.randint(0, 512 - crop_h), random.randint(0, 512 - crop_w)
                x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

                x_tgt_renorm = self.t_vgg_renorm(gt_img * 0.5 + 0.5)
                x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                gram_loss = get_gram_loss(x_tgt_pred_renorm.float(), x_tgt_renorm.float(), self.net_vgg)
            else:
                gram_loss = torch.tensor(0.0).to(self.weight_dtype)

        logits_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

        if prog_si >= 0:  # in progressive training
            bg, ed = self.begin_ends[prog_si]
            assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
        else:  # not in progressive training
            lw = self.loss_weight
        logits_loss = logits_loss.mean(dim=-1).mean()

        loss = logits_loss + diff_loss * 2.0

        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        if it == 0 or it in metric_lg.log_iters:
            pred_BL = logits_BLV.data.argmax(dim=-1)
            Lmean = self.val_loss(logits_BLV.contiguous().reshape(-1, V), gt_BL.reshape(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            if prog_si >= 0:    # in progressive training
                Ltail = acc_tail = -1
            else:  # not in progressive training
                Ltail = self.val_loss(logits_BLV.contiguous().data[:, self.start_token:].reshape(-1, V), gt_BL[:, self.start_token:].reshape(-1)).item()
                acc_tail = (pred_BL[:, self.start_token:] == gt_BL[:, self.start_token:]).float().mean().item() * 100
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm)
            metric_lg.update(l2=l2_loss.item(), lpips=lpips_loss.item(), gram=gram_loss.item(), tnm=grad_norm.item())

        # log to tensorboard
        if self.wandb_flag:
            if dist.is_master():
                log_dict = {
                    "train/loss": loss.item(),
                    "train/logits_loss": logits_loss.item(),
                    "train/diff_loss": diff_loss.item(),
                    "train/l2_loss": l2_loss.item(),
                    "train/lpips_loss": lpips_loss.item(),
                    "train/gram_loss": gram_loss.item(),
                    "train/grad_norm": grad_norm.item(),
                    "train/lr": lr, "train/wd": wd
                }
                if it % 5000 == 0:
                    log_dict.update(self.get_img_log_dict(lr_inp, gt_inp, pred, stage="train"))

                wandb.log(log_dict)

        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    # step-by-step inference
    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        l2_loss, lpips_loss, gram_loss = 0, 0, 0
        log_dict = {"eval/input": [], "eval/gt": [], "eval/output": []}
        stt = time.time()
        training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for step, batch in enumerate(ld_val):
            if step >= self.args.num_samples_eval:
                break
            gt_inp = batch["pixel_values"].to(dist.get_device(), non_blocking=True)
            lr_inp = batch["conditioning_pixel_values"].to(dist.get_device(), non_blocking=True)
            # torchvision.utils.save_image(
            #     tensor=gt_inp[0],
            #     fp="dbg/eval_gt.png",
            #     normalize=True,
            #     nrow=2  # Arrange images in a grid with 4 columns
            # )
            #
            # torchvision.utils.save_image(
            #     tensor=lr_inp[0],
            #     fp="dbg/eval_lr.png",
            #     normalize=True,
            #     nrow=2
            # )
            label_B = batch["label_B"].to(dist.get_device(), non_blocking=True)
            B, _, C, H, W = gt_inp.shape
            assert B == 1, "Use batch size 1 for eval."
            gt_inp = rearrange(gt_inp, 'b v c h w -> (b v) c h w')
            lr_inp = rearrange(lr_inp, 'b v c h w -> (b v) c h w')
            label_B = rearrange(label_B, 'b v -> (b v)')

            # [-1, 1]
            pred = self.var_wo_ddp.autoregressive_infer_cfg(B=B * self.view_num, cfg=1.0, top_k=1, top_p=0.96,
                                                                  text_hidden=None, lr_inp=lr_inp,
                                                                  negative_text=None, label_B=label_B, tile_flag=False,
                                                                  lr_inp_scale=None, more_smooth=False, normalize=True)

            if self.wandb_flag and dist.is_master() and step % 10 == 0:
                for key, list in self.get_img_log_dict(lr_inp, gt_inp, pred, stage="eval").items():
                    log_dict[key].extend(list)

            # rm ref view img, only eval render view img
            pred_img = rearrange(pred, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
            gt_img = rearrange(gt_inp, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]

            # Reconstruction loss
            l2_loss += F.mse_loss(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype), reduction="mean")
            lpips_loss += self.net_lpips(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype)).mean()

            x_tgt_pred_renorm = self.t_vgg_renorm(pred_img * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
            crop_h, crop_w = 400, 400
            top, left = random.randint(0, 512 - crop_h), random.randint(0, 512 - crop_w)
            x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

            x_tgt_renorm = self.t_vgg_renorm(gt_img * 0.5 + 0.5)
            x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

            gram_loss += get_gram_loss(x_tgt_pred_renorm.float(), x_tgt_renorm.float(), self.net_vgg)

            tot += B
        # report img
        if dist.is_master():
            wandb.log(log_dict)

        self.var_wo_ddp.train(training)

        stats = l2_loss.new_tensor([l2_loss.item(), lpips_loss.item(), gram_loss.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        l2_loss, lpips_loss, gram_loss, _ = stats.tolist()
        return l2_loss, lpips_loss, gram_loss, tot, time.time() - stt

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                if k == 'var_wo_ddp' and self.args.use_lora:
                    state[k] = lora.lora_state_dict(m)
                else:
                    state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                if k == 'var_wo_ddp' and self.args.use_lora:
                    # assuming loading lora
                    ret = m.load_state_dict(state[k], strict=False)
                else:
                    ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)

class VAETrainer(BASETrainer):
    def __init__(
            self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
            vae_local: VQVAE, var_wo_ddp, vae: DDP,
            text_encoder, clip_vision, exp_name,
            vae_opt: AmpOptimizer, label_smooth: float, wandb_flag=False,
            fp16: int = 1,
            args=None,
    ):
        super().__init__(
            device=device, patch_nums=patch_nums, resos=resos,
            vae_local=vae_local, var_wo_ddp=var_wo_ddp,
            text_encoder=text_encoder, clip_vision=clip_vision, exp_name=exp_name, label_smooth=label_smooth,
            wandb_flag=wandb_flag, fp16=fp16, args=args
        )
        self.vae = vae
        self.vae_opt = vae_opt
        self.args = args

    @torch.no_grad()
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        l2_loss, lpips_loss, gram_loss = 0, 0, 0
        log_dict = {"eval/input": [], "eval/gt": [], "eval/output": []}
        stt = time.time()
        training = self.vae_local.training
        self.vae_local.eval()
        for step, batch in enumerate(ld_val):
            if step >= self.args.num_samples_eval:
                break
            gt_inp = batch["pixel_values"].to(dist.get_device(), non_blocking=True)
            lr_inp = batch["conditioning_pixel_values"].to(dist.get_device(), non_blocking=True)
            # torchvision.utils.save_image(
            #     tensor=gt_inp[0],
            #     fp="dbg/eval_gt.png",
            #     normalize=True,
            #     nrow=2  # Arrange images in a grid with 4 columns
            # )
            #
            # torchvision.utils.save_image(
            #     tensor=lr_inp[0],
            #     fp="dbg/eval_lr.png",
            #     normalize=True,
            #     nrow=2
            # )
            label_B = batch["label_B"].to(dist.get_device(), non_blocking=True)
            B, _, C, H, W = gt_inp.shape
            assert B == 1, "Use batch size 1 for eval."
            gt_inp = rearrange(gt_inp, 'b v c h w -> (b v) c h w')
            lr_inp = rearrange(lr_inp, 'b v c h w -> (b v) c h w')
            label_B = rearrange(label_B, 'b v -> (b v)')

            # [-1, 1]
            pred = self.var_wo_ddp.autoregressive_infer_cfg(B=B * self.view_num, cfg=1.0, top_k=1, top_p=0.96,
                                                                  text_hidden=None, lr_inp=lr_inp,
                                                                  negative_text=None, label_B=label_B, tile_flag=False,
                                                                  lr_inp_scale=None, more_smooth=False, normalize=True)

            if self.wandb_flag and dist.is_master() and step % 10 == 0:
                for key, list in self.get_img_log_dict(lr_inp, gt_inp, pred, stage="eval").items():
                    log_dict[key].extend(list)

            # rm ref view img, only eval render view img
            pred_img = rearrange(pred, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
            gt_img = rearrange(gt_inp, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]

            # Reconstruction loss
            l2_loss += F.mse_loss(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype), reduction="mean")
            lpips_loss += self.net_lpips(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype)).mean()

            x_tgt_pred_renorm = self.t_vgg_renorm(pred_img * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
            crop_h, crop_w = 400, 400
            top, left = random.randint(0, 512 - crop_h), random.randint(0, 512 - crop_w)
            x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

            x_tgt_renorm = self.t_vgg_renorm(gt_img * 0.5 + 0.5)
            x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

            gram_loss += get_gram_loss(x_tgt_pred_renorm.float(), x_tgt_renorm.float(), self.net_vgg)

            tot += B
        # report img
        if dist.is_master():
            wandb.log(log_dict)

        self.vae_local.train(training)

        stats = l2_loss.new_tensor([l2_loss.item(), lpips_loss.item(), gram_loss.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        l2_loss, lpips_loss, gram_loss, _ = stats.tolist()
        return l2_loss, lpips_loss, gram_loss, tot, time.time() - stt

    def train_step(
            self, it: int, g_it: int, stepping: bool, metric_lg: MetricLogger,
            gt_inp: FTen, lr_inp: Union[ITen, FTen], label_B,
            text, prog_si: int, prog_wp_it: float, lr, wd,
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # if progressive training
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1  # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1  # max prog, as if no prog

        # forward
        B, V = gt_inp.shape[0], self.vae_local.vocab_size
        self.vae.require_backward_grad_sync = stepping

        with torch.no_grad():
            cond_features = self.vae_local.get_cond_features(lr_inp)
            # w ref inference
            f_hat = self.var_wo_ddp.autoregressive_infer_cfg(B=B, cfg=1.0, top_k=1, top_p=0.96,
                                                             text_hidden=None, lr_inp=lr_inp,
                                                             negative_text=None, label_B=label_B, tile_flag=True,
                                                             lr_inp_scale=None, more_smooth=False)
            for i, cond_feature in enumerate(cond_features):
                cond_features[i] = cond_feature.detach()
            # remove ref view
            f_hat = f_hat.detach()
            torch.cuda.empty_cache()

        with self.vae_opt.amp_ctx:
            pred = self.vae_local.fhat_to_img(f_hat, cond_features=[cond_features])  # [-1, 1]

        if prog_si >= 0:  # in progressive training
            bg, ed = self.begin_ends[prog_si]
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
        else:  # not in progressive training
            lw = self.loss_weight

        if not self.args.use_ref_view_loss:
            # remove ref view
            gt_img = rearrange(gt_inp, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
            pred_img = rearrange(pred, '(b v) c h w -> b v c h w', v=self.view_num)[:, 0]
        else:
            gt_img = gt_inp
            pred_img = pred

        # Reconstruction loss
        l2_loss = F.mse_loss(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype), reduction="mean")
        lpips_loss = self.net_lpips(pred_img.to(self.weight_dtype), gt_img.to(self.weight_dtype)).mean()

        with torch.cuda.amp.autocast(enabled=False):
            if self.prog_it > prog_wp_it and self.args.use_gram_loss:
                x_tgt_pred_renorm = self.t_vgg_renorm(pred_img * 0.5 + 0.5)  # [-1, 1] -> [0, 1]
                crop_h, crop_w = 400, 400
                top, left = random.randint(0, 512 - crop_h), random.randint(0, 512 - crop_w)
                x_tgt_pred_renorm = crop(x_tgt_pred_renorm, top, left, crop_h, crop_w)

                x_tgt_renorm = self.t_vgg_renorm(gt_img * 0.5 + 0.5)
                x_tgt_renorm = crop(x_tgt_renorm, top, left, crop_h, crop_w)

                gram_loss = get_gram_loss(x_tgt_pred_renorm.float(), x_tgt_renorm.float(), self.net_vgg)
            else:
                gram_loss = torch.tensor(0.0).to(self.weight_dtype)
            loss = l2_loss * 1.0 + lpips_loss * 1.0 + gram_loss * 0.5  # TODO not matching?

        # backward
        grad_norm, scale_log2 = self.vae_opt.backward_clip_step(loss=loss, stepping=stepping)

        # log
        if it == 0 or it in metric_lg.log_iters:
            metric_lg.update(l2_loss=l2_loss.item(), lpips_loss=lpips_loss.item(), gram_loss=gram_loss.item(), tnm=grad_norm.item())

        # log to tensorboard
        if self.wandb_flag:
            if dist.is_master():
                log_dict = {"train/loss": loss.item(),
                           "train/l2_loss": l2_loss.item(), "train/lpips_loss": lpips_loss.item(),
                           "train/gram_loss": gram_loss.item(),
                           "train/grad_norm": grad_norm, "train/lr": lr, "train/wd": wd}

                if it % 100 == 0:
                    log_dict.update(self.get_img_log_dict(lr_inp, gt_inp, pred, stage="train"))

                wandb.log(log_dict)

        self.vae_local.quantize.prog_si = -1
        return grad_norm, scale_log2

    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state

    def load_state_dict(self, state, strict=True):
        for k in ('var_wo_ddp', 'vae_local', 'vae_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VAETrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VAETrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAE.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)

def from_pretrained_orig(var, state_dict):
    for k, v in var.state_dict().items():
        if '.cross_attn.' in k:
            if 'mat_q' in k:
                key = k.replace(".cross_attn", ".attn").replace("mat_q", "mat_qkv")
                state_dict[k] = state_dict[key][0:state_dict[key].shape[0] // 3, :]
            elif 'mat_kv' in k:
                key = k.replace(".cross_attn", ".attn").replace("mat_kv", "mat_qkv")
                state_dict[k] = state_dict[key][state_dict[key].shape[0] // 3:, :v.shape[1]]
            else:
                key = k.replace(".cross_attn", ".attn")
                state_dict[k] = state_dict[key]
        elif 'class_emb' in k:
            value = state_dict[k]
            if value.shape[0] > v.shape[0]:
                state_dict[k] = state_dict[k][:v.shape[0], :]
            elif value.shape[0] < v.shape[0]:
                state_dict[k] = torch.cat((state_dict[k][:3830, :], state_dict[k][:3830, :]), dim=0)
    for key, value in var.state_dict().items():
        if key in state_dict and state_dict[key].shape != value.shape:
            print(key)
            state_dict.pop(key)
    ret = var.load_state_dict(state_dict, strict=False)
    missing, unexpected = ret
    print(f'[VAR.load_state_dict] missing:  {missing}')
    print(f'[VAR.load_state_dict] unexpected:  {unexpected}')
    del state_dict

    return var