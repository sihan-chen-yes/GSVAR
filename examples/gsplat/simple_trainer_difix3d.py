import json
import math
import os
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
import random
from PIL import Image
from copy import deepcopy

from torchvision.transforms.functional import to_tensor

from examples.gsplat.datasets.colmap import Dataset, Parser
from examples.gsplat.datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from examples.gsplat.gsplat_utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from examples.gsplat.lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.optimizers import SelectiveAdam

from examples.utils import CameraPoseInterpolator
from models import VAR_RoPE, VQVAE, build_var
from utils import arg_util
import dist
from einops import rearrange, repeat
from utils.misc import metrics, save_img, gaussian_weights, pt_to_numpy, numpy_to_pil
from torchvision.transforms import InterpolationMode, transforms
from utils.visual_util import images_to_video
from difix.src.pipeline_difix import DifixPipeline
from difix.src.model import Difix

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = True # ! turn off viser
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 60_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [1_0000, 2_0000, 3_0000, 3_5000, 4_0000, 4_5000, 5_0000, 5_5000, 6_0000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [1_0000, 2_0000, 3_0000, 4_0000, 4_5000, 5_0000, 5_5000, 6_0000])
    # Steps to fix the artifacts
    fix: bool = True
    fixer: str = "gsvar"
    fix_steps: List[int] = field(default_factory=lambda: [3_000, 6_000, 8_000, 10_000, 12_000, 14_000, 16_000, 18_000, 20_000, 22_000, 24_000, 26_000, 28_000, 30_000, 32_000, 34_000, 36_000, 38_000, 40_000, 42_000, 44_000, 46_000, 48_000, 50_000, 52_000, 54_000, 56_000, 58_000, 60_000])

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2
    # Weight for iterative 3d update
    novel_data_lambda: float = 0.3

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # save_video
    save_video: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

    # args for gsvar
    vae_model_path: str = 'checkpoints/VQVAE.pth'
    var_test_path: str = "checkpoints/VARSR.pth"
    # use ref img
    use_ref: bool = True
    # use VAE bridge
    use_bridge: bool = False
    depth: int = 24     # VAR depth
    tiling: bool = False


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 / 10 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3 / 5),
        ("quats", torch.nn.Parameter(quats), 1e-3 / 5),
        ("opacities", torch.nn.Parameter(opacities), 5e-2 / 5),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3 / 50))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20 / 50))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"
        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )
            
        # Fixer trajectory 
        self.interpolator = CameraPoseInterpolator(rotation_weight=1.0, translation_weight=1.0)

        self.current_novel_poses = self.parser.camtoworlds[self.trainset.indices]
        self.current_parser = self.parser

        self.novelloaders = []
        self.novelloaders_iter = []
        
        # GSVAR fixer
        if cfg.fix:
            if cfg.fixer == "gsvar":

                self.gsvar_args: arg_util.Args = arg_util.get_args(default=True)

                self.gsvar_args.var_test_path = cfg.var_test_path
                self.gsvar_args.use_ref = cfg.use_ref
                self.gsvar_args.use_bridge = cfg.use_bridge
                self.gsvar_args.depth = cfg.depth
                self.fixer = self.load_gsvar(self.gsvar_args)
            elif cfg.fixer == "difix":
                self.fixer = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
                self.fixer.set_progress_bar_config(disable=True)
                self.fixer.to("cuda")
            elif cfg.fixer == "difix_re":
                self.fixer = Difix(
                    pretrained_path="difix/checkpoints/model_50001.pkl",
                    timestep=199,
                    mv_unet=True, # w ref
                )

    def load_gsvar(self, args):
        vae_ckpt = args.vae_model_path
        var_ckpt = args.var_test_path

        vae, var = build_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4, controlnet_depth=args.depth,
            # hard-coded VQVAE hyperparameters
            device=dist.get_device(), patch_nums=args.patch_nums, control_patch_nums=args.patch_nums,
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
        return var

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self, step=0):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = step

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            if len(self.novelloaders) == 0 or random.random() < 0.7:
                try:
                    data = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(trainloader)
                    data = next(trainloader_iter)
                is_novel_data = False
            else:
                try:
                    data = next(self.novelloaders_iter[-1])
                except StopIteration:
                    self.novelloaders_iter[-1] = iter(self.novelloaders[-1])
                    data = next(self.novelloaders_iter[-1])        
                is_novel_data = True

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            alpha_masks = data["alpha_mask"].to(device) if "alpha_mask" in data else None  # [1, H, W, 1]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            if is_novel_data and alpha_masks is not None:
                colors = colors * (alpha_masks > 0.5).float()
                pixels = pixels * (alpha_masks > 0.5).float()

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss = (
                    loss
                    + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            if is_novel_data:
                loss = loss * cfg.novel_data_lambda
            else:
                loss = loss * 1.5
            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

            # run fixer
            if cfg.fix and step in [i - 1 for i in cfg.fix_steps]:
                self.fix(step)
            
            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def enhance(self, render_view, ref_view):
        """

        Args:
            render_view:
            ref_view:

        Returns:
            enhanced_render_view_
        """
        if self.cfg.fixer == "gsvar":
            if not self.cfg.tiling:
                img_preproc = transforms.Compose([
                    # forcing to 512
                    transforms.Resize((512, 512), interpolation=InterpolationMode.LANCZOS),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])

                inp = torch.stack([img_preproc(render_view), img_preproc(ref_view)], dim=0).to(self.gsvar_args.device, non_blocking=True)
                label_B = torch.tensor([0, 1], dtype=torch.long).to(self.gsvar_args.device, non_blocking=True)
                # [0, 1]
                output_image = self.fixer.autoregressive_infer_cfg(B=2, cfg=1.0, top_k=1, top_p=0.96,
                                                                   text_hidden=None, lr_inp=inp, negative_text=None, label_B=label_B, lr_inp_scale = None,
                                                                   more_smooth=False)
                output_image = rearrange(output_image, '(b v) c h w -> b v c h w', v=2)[:, 0]
                output_image = numpy_to_pil(pt_to_numpy(output_image))[0]
            else:
                img_preproc = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])

                rscale = 1
                ori_h = render_view.size[0] * rscale
                ori_w = render_view.size[1] * rscale

                # dynamic resize to 16x
                lr_inp = render_view.resize((max(math.ceil(render_view.size[0] / 16) * 16 * rscale, 512),
                                        max(math.ceil(render_view.size[1] / 16) * 16 * rscale, 512)))

                ref_inp = ref_view.resize((max(math.ceil(ref_view.size[0] / 16) * 16 * rscale, 512),
                                          max(math.ceil(ref_view.size[1] / 16) * 16 * rscale, 512)))

                inp = torch.stack([img_preproc(lr_inp), img_preproc(ref_inp)], dim=0).to(self.gsvar_args.device, non_blocking=True)
                label_B = torch.tensor([0, 1], dtype=torch.long).to(self.gsvar_args.device, non_blocking=True)

                B = inp.shape[0]
                h, w = math.ceil(inp.shape[2] / 16), math.ceil(inp.shape[3] / 16)
                tile_size = 32
                tile_overlap = 8
                tile_weights = gaussian_weights(32, 32, 1)

                grid_rows = 0
                cur_x = 0
                while cur_x < h:
                    cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                    grid_rows += 1

                grid_cols = 0
                cur_y = 0
                while cur_y < w:
                    cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                    grid_cols += 1
                recon_pred = []
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if col < grid_cols - 1 or row < grid_rows - 1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size - tile_overlap * row, 0)
                            ofs_y = max(col * tile_size - tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows - 1:
                            ofs_x = h - tile_size
                        if col == grid_cols - 1:
                            ofs_y = w - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size
                        # input tile dimensions
                        inp_tile = inp[:, :, input_start_x * 16:input_end_x * 16, input_start_y * 16:input_end_y * 16]

                        with torch.inference_mode():
                            V = 2 if self.gsvar_args.use_ref else 1
                            with torch.autocast('cuda', enabled=True, dtype=torch.float32,
                                                cache_enabled=True):  # using bfloat16 can be faster
                                output = self.fixer.autoregressive_infer_cfg(B=V, cfg=1.0, top_k=1, top_p=0.96,
                                                                      text_hidden=None, lr_inp=inp_tile, negative_text=None,
                                                                      label_B=label_B, lr_inp_scale=None,
                                                                      tile_flag=True,
                                                                      more_smooth=False)
                                output = rearrange(output, '(b v) c h w -> b v c h w', v=V)
                                # rm ref img
                                recon_B3HW = output[:, 0]
                                recon_pred.append(recon_B3HW)

                preds = torch.zeros((B, 32, h, w), device=inp.device)
                contributors = torch.zeros((B, 32, h, w), device=inp.device)
                # Add each tile contribution to overall latents
                for row in range(grid_rows):
                    for col in range(grid_cols):
                        if col < grid_cols - 1 or row < grid_rows - 1:
                            # extract tile from input image
                            ofs_x = max(row * tile_size - tile_overlap * row, 0)
                            ofs_y = max(col * tile_size - tile_overlap * col, 0)
                            # input tile area on total image
                        if row == grid_rows - 1:
                            ofs_x = h - tile_size
                        if col == grid_cols - 1:
                            ofs_y = w - tile_size

                        input_start_x = ofs_x
                        input_end_x = ofs_x + tile_size
                        input_start_y = ofs_y
                        input_end_y = ofs_y + tile_size

                        preds[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += recon_pred[row * grid_cols + col] * tile_weights  # [b, c, h, w]
                        contributors[:, :, input_start_x:input_end_x, input_start_y:input_end_y] += tile_weights
                # Average overlapping areas with more than 1 contributor
                preds /= contributors
                with torch.no_grad():
                    cond_features = [self.fixer.vae_proxy[0].get_cond_features(lr_inp)] if self.gsvar_args.use_bridge else None
                    output_image = self.fixer.vae_proxy[0].fhat_to_img(preds, cond_features).add_(1).mul_(0.5)
                output_image = numpy_to_pil(pt_to_numpy(output_image))[0]

            output_image = output_image.resize(render_view.size, Image.LANCZOS)

        elif self.cfg.fixer == "difix":
            output_image = self.fixer(
                "remove degradation",
                image=render_view,
                ref_image=ref_view,
                num_inference_steps=1,
                timesteps=[199],
                guidance_scale=0.0
            ).images[0]
            output_image = output_image.resize(render_view.size, resample=Image.BICUBIC)

        elif self.cfg.fixer ==  "difix_re":
            output_image = self.fixer.sample(
                render_view,
                height=576,
                width=1024,
                ref_image=ref_view,
                prompt="remove degradation"
            )

        return output_image


    @torch.no_grad()
    def fix(self, step: int):
        print("Running fixer...")
        if len(self.cfg.fix_steps) == 1:
            novel_poses = self.parser.camtoworlds[self.valset.indices]
        else:
            novel_poses = self.interpolator.shift_poses(self.current_novel_poses, self.parser.camtoworlds[self.valset.indices], distance=0.5)
        
        self.render_traj(step, novel_poses)
        image_paths = [f"{self.render_dir}/novel/{step}/Pred/{i:04d}.png" for i in range(len(novel_poses))]

        if len(self.novelloaders) == 0:
            ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
            ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]
        else:
            ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], novel_poses)
            ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]
        assert len(image_paths) == len(ref_image_paths) == len(novel_poses)

        for i in tqdm.trange(0, len(novel_poses), desc="Fixing artifacts..."):
            image = Image.open(image_paths[i]).convert("RGB")
            ref_image = Image.open(ref_image_paths[i]).convert("RGB")

            output_image = self.enhance(render_view=image, ref_view=ref_image)

            os.makedirs(f"{self.render_dir}/novel/{step}/Fixed", exist_ok=True)
            output_image.save(f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png")
            if ref_image is not None:
                os.makedirs(f"{self.render_dir}/novel/{step}/Ref", exist_ok=True)
                ref_image.save(f"{self.render_dir}/novel/{step}/Ref/{i:04d}.png")
    
        parser = deepcopy(self.parser)
        parser.test_every = 0
        parser.image_paths = [f"{self.render_dir}/novel/{step}/Fixed/{i:04d}.png" for i in range(len(novel_poses))]
        parser.image_names = [os.path.basename(p) for p in parser.image_paths]
        parser.alpha_mask_paths = [f"{self.render_dir}/novel/{step}/Alpha/{i:04d}.png" for i in range(len(novel_poses))]
        parser.camtoworlds = novel_poses
        parser.camera_ids = [parser.camera_ids[0]] * len(novel_poses)
        
        print(f"Adding {len(parser.image_paths)} fixed images to novel dataset...")
        dataset = Dataset(parser, split="train")
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        self.novelloaders.append(dataloader)
        self.novelloaders_iter.append(iter(dataloader))

        self.current_novel_poses = novel_poses

        # save video for visualization
        if self.cfg.save_video:
            images_to_video(image_dir=f"{self.render_dir}/novel/{step}/Fixed",
                            output_file=f"fixed_{step}.mp4",
                            fps=15)
            images_to_video(image_dir=f"{self.render_dir}/novel/{step}/Pred",
                            output_file=f"pred_{step}.mp4",
                            fps=15)
            images_to_video(image_dir=f"{self.render_dir}/novel/{step}/Ref",
                            output_file=f"ref_{step}.mp4",
                            fps=15)

            
    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        ref_image_indices = self.interpolator.find_nearest_assignments(self.parser.camtoworlds[self.trainset.indices], self.parser.camtoworlds[self.valset.indices])
        ref_image_paths = [self.parser.image_paths[i] for i in np.array(self.trainset.indices)[ref_image_indices]]

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm.tqdm(valloader)):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            colors, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                pixels_path = f"{self.render_dir}/val/{step}/GT/{i:04d}.png"
                os.makedirs(os.path.dirname(pixels_path), exist_ok=True)
                pixels_canvas = pixels.squeeze(0).cpu().numpy()
                pixels_canvas = (pixels_canvas * 255).astype(np.uint8)
                imageio.imwrite(pixels_path, pixels_canvas)

                colors_path = f"{self.render_dir}/val/{step}/Pred/{i:04d}.png"
                os.makedirs(os.path.dirname(colors_path), exist_ok=True)
                colors_canvas = colors.squeeze(0).cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)

                if self.cfg.fix:
                    # real-time post-processing via fixer
                    image = Image.open(colors_path).convert("RGB")
                    ref_image = Image.open(ref_image_paths[i]).convert("RGB")

                    output_image = self.enhance(render_view=image, ref_view=ref_image)

                    os.makedirs(f"{self.render_dir}/val/{step}/Post", exist_ok=True)
                    output_image.save(f"{self.render_dir}/val/{step}/Post/{i:04d}.png")
                    if ref_image is not None:
                        os.makedirs(f"{self.render_dir}/val/{step}/Ref", exist_ok=True)
                        ref_image.save(f"{self.render_dir}/val/{step}/Ref/{i:04d}.png")

                alphas_path = f"{self.render_dir}/val/{step}/Alpha/{i:04d}.png"
                os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
                alphas_canvas = (alphas < 0.5).squeeze(0).cpu().numpy()
                alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
                Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 3, H, W] gt
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 3, H, W] render
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if self.cfg.fix:
                    fixed_p = to_tensor(output_image)[None].to(device)  # [1, 3, H, W] fixed render
                    metrics["post_psnr"].append(self.psnr(fixed_p, pixels_p))
                    metrics["post_ssim"].append(self.ssim(fixed_p, pixels_p))
                    metrics["post_lpips"].append(self.lpips(fixed_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            base_msg = (
                f"PSNR: {stats['psnr']:.3f}\n"
                f"SSIM: {stats['ssim']:.4f}\n"
                f"LPIPS: {stats['lpips']:.3f}\n"
            )

            if self.cfg.fix:
                base_msg += (
                    f"POST_PSNR: {stats.get('post_psnr', 0):.3f}\n"
                    f"POST_SSIM: {stats.get('post_ssim', 0):.4f}\n"
                    f"POST_LPIPS: {stats.get('post_lpips', 0):.3f}\n"
                )

            base_msg += (
                f"Time: {stats['ellipse_time']:.3f}s/image\n"
                f"Number of GS: {stats['num_GS']}"
            )

            print(base_msg)
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

            # save video for visualization
            if self.cfg.save_video:
                images_to_video(image_dir=f"{self.render_dir}/val/{step}/GT",
                                output_file=f"gt_{step}.mp4",
                                fps=15)
                images_to_video(image_dir=f"{self.render_dir}/val/{step}/Pred",
                                output_file=f"pred_{step}.mp4",
                                fps=15)
                if self.cfg.fix:
                    images_to_video(image_dir=f"{self.render_dir}/val/{step}/Post",
                                    output_file=f"post_{step}.mp4",
                                    fps=15)
                    images_to_video(image_dir=f"{self.render_dir}/val/{step}/Ref",
                                    output_file=f"ref_{step}.mp4",
                                    fps=15)

    @torch.no_grad()
    def render_traj(self, step: int, camtoworlds_all=None, batch_size=8, tag="novel"):
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if camtoworlds_all is None:
            camtoworlds_all = self.parser.camtoworlds[5:-5]
            if cfg.render_traj_path == "interp":
                camtoworlds_all = generate_interpolated_path(
                    camtoworlds_all, 1
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "ellipse":
                height = camtoworlds_all[:, 2, 3].mean()
                camtoworlds_all = generate_ellipse_path_z(
                    camtoworlds_all, height=height
                )  # [N, 3, 4]
            elif cfg.render_traj_path == "spiral":
                camtoworlds_all = generate_spiral_path(
                    camtoworlds_all,
                    bounds=self.parser.bounds * self.scene_scale,
                    spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
                )
            else:
                raise ValueError(
                    f"Render trajectory type not supported: {cfg.render_traj_path}"
                )

            camtoworlds_all = np.concatenate(
                [
                    camtoworlds_all,
                    np.repeat(
                        np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                    ),
                ],
                axis=1,
            )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        for i in tqdm.trange(0, len(camtoworlds_all), batch_size, desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + batch_size]
            Ks = K[None].repeat(camtoworlds.shape[0], 1, 1)

            renders, alphas, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [B, H, W, 4]

            for j in range(renders.shape[0]):
                colors = torch.clamp(renders[j, ..., 0:3], 0.0, 1.0)  # [H, W, 3]
                depths = renders[j, ..., 3:4]  # [H, W, 1]
                depths = (depths - depths.min()) / (depths.max() - depths.min())
                
                idx = i + j
                colors_path = f"{self.render_dir}/{tag}/{step}/Pred/{idx:04d}.png"
                os.makedirs(os.path.dirname(colors_path), exist_ok=True)
                colors_canvas = colors.cpu().numpy()
                colors_canvas = (colors_canvas * 255).astype(np.uint8)
                imageio.imwrite(colors_path, colors_canvas)
                
                alphas_path = f"{self.render_dir}/{tag}/{step}/Alpha/{idx:04d}.png"
                os.makedirs(os.path.dirname(alphas_path), exist_ok=True)
                alphas_canvas = alphas[j].float().cpu().numpy()
                alphas_canvas = (alphas_canvas * 255).astype(np.uint8)
                Image.fromarray(alphas_canvas.squeeze(), mode='L').save(alphas_path)

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.train(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)
