import datetime
import functools
import glob
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import Iterator, List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist

import dist
from utils import arg_util
from PIL import Image
import pyiqa
from torchvision import transforms
from skimage import io

os_system = functools.partial(subprocess.call, shell=True)
def echo(info):
    os_system(f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"')
def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f'[fetch free_port file] timeout cnt={cnt}')
        else:
            return sp.stdout.decode('utf-8'), sp.stderr.decode('utf-8')


def time_str(fmt='[%m-%d %H:%M:%S]'):
    return datetime.datetime.now(tz=pytz.timezone('Asia/Shanghai')).strftime(fmt)


def init_distributed_mode(local_out_path, only_sync_master=False, timeout=30):
    try:
        dist.initialize(fork=False, timeout=timeout)
        dist.barrier()
    except RuntimeError:
        print(f'{">"*75}  NCCL Error  {"<"*75}', flush=True)
        time.sleep(10)
    
    if local_out_path is not None: os.makedirs(local_out_path, exist_ok=True)
    _change_builtin_print(dist.is_local_master())
    if (dist.is_master() if only_sync_master else dist.is_local_master()) and local_out_path is not None and len(local_out_path):
        sys.stdout, sys.stderr = SyncPrint(local_out_path, sync_stdout=True), SyncPrint(local_out_path, sync_stdout=False)


def _change_builtin_print(is_master):
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    if type(builtin_print) != type(open):
        return
    
    def prt(*args, **kwargs):
        force = kwargs.pop('force', False)
        clean = kwargs.pop('clean', False)
        deeper = kwargs.pop('deeper', False)
        if is_master or force:
            if not clean:
                f_back = sys._getframe().f_back
                if deeper and f_back.f_back is not None:
                    f_back = f_back.f_back
                file_desc = f'{f_back.f_code.co_filename:24s}'[-24:]
                builtin_print(f'{time_str()} ({file_desc}, line{f_back.f_lineno:-4d})=>', *args, **kwargs)
            else:
                builtin_print(*args, **kwargs)
    
    __builtin__.print = prt


class SyncPrint(object):
    def __init__(self, local_output_dir, sync_stdout=True):
        self.sync_stdout = sync_stdout
        self.terminal_stream = sys.stdout if sync_stdout else sys.stderr
        fname = os.path.join(local_output_dir, 'stdout.txt' if sync_stdout else 'stderr.txt')
        existing = os.path.exists(fname)
        self.file_stream = open(fname, 'a')
        if existing:
            self.file_stream.write('\n'*7 + '='*55 + f'   RESTART {time_str()}   ' + '='*55 + '\n')
        self.file_stream.flush()
        self.enabled = True
    
    def write(self, message):
        self.terminal_stream.write(message)
        self.file_stream.write(message)
    
    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()
    
    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        self.file_stream.flush()
        self.file_stream.close()
        if self.sync_stdout:
            sys.stdout = self.terminal_stream
            sys.stdout.flush()
        else:
            sys.stderr = self.terminal_stream
            sys.stderr.flush()

    def isatty(self):
        return False

    def __del__(self):
        self.close()


class DistLogger(object):
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose
    
    @staticmethod
    def do_nothing(*args, **kwargs):
        pass
    
    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing


class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try: import tensorflow_io as tfio
        except: pass
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0
    
    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            # assert isinstance(v, (float, int)), type(v)
            if step is None:  # iter wise
                it = self.step
                if it == 0 or (it + 1) % 500 == 0:
                    if hasattr(v, 'item'): v = v.item()
                    self.writer.add_scalar(f'{head}/{k}', v, it)
            else:  # epoch wise
                if hasattr(v, 'item'): v = v.item()
                self.writer.add_scalar(f'{head}/{k}', v, step)
    
    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            try:
                self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
            except Exception as e:
                print(f'[log_tensor_as_distri writer.add_histogram failed]: {e}')
    
    def log_image(self, tag, img_chw, step=None):
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            self.writer.add_image(tag, img_chw, step, dataformats='CHW')
    
    def flush(self):
        self.writer.flush()
    
    def close(self):
        self.writer.close()


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """
    
    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
    
    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
    
    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.allreduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]
    
    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0
    
    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)
    
    @property
    def global_avg(self):
        return self.total / (self.count or 1)
    
    @property
    def max(self):
        return max(self.deque)
    
    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0
    
    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return remain_secs, str(datetime.timedelta(seconds=round(remain_secs))), time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs))
    
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter='  '):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.iter_end_t = time.time()
        self.log_iters = []
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if hasattr(v, 'item'): v = v.item()
            # assert isinstance(v, (float, int)), type(v)
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    
    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))
    
    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            if len(meter.deque):
                loss_str.append(
                    "{}: {}".format(name, str(meter))
                )
        return self.delimiter.join(loss_str)
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()
    
    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def log_every(self, start_it, max_iters, itrt, print_freq, header=None):
        self.log_iters = set(np.linspace(0, max_iters-1, print_freq, dtype=int).tolist())
        self.log_iters.add(start_it)
        if not header:
            header = ''
        start_time = time.time()
        self.iter_end_t = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.4f}')
        self.data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(max_iters))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)
        
        if isinstance(itrt, Iterator) and not hasattr(itrt, 'preload') and not hasattr(itrt, 'set_epoch'):
            for i in range(start_it, max_iters):
                obj = next(itrt)
                self.data_time.update(time.time() - self.iter_end_t)
                yield i, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if i in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()
        else:
            if isinstance(itrt, int): itrt = range(itrt)
            for i, obj in enumerate(itrt):
                self.data_time.update(time.time() - self.iter_end_t)
                yield i, obj
                self.iter_time.update(time.time() - self.iter_end_t)
                if i in self.log_iters:
                    eta_seconds = self.iter_time.global_avg * (max_iters - i)
                    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                    print(log_msg.format(
                        i, max_iters, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time)), flush=True)
                self.iter_end_t = time.time()
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{}   Total time:      {}   ({:.3f} s / it)'.format(
            header, total_time_str, total_time / max_iters), flush=True)


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True)


def auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, dict, dict]:
    info = []
    file = os.path.join(args.local_out_dir_path, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f'[auto_resume] no ckpt found @ {file}')
        info.append(f'[auto_resume quit]')
        return info, 0, 0, {}, {}
    else:
        info.append(f'[auto_resume] load ckpt from @ {all_ckpt[0]} ...')
        ckpt = torch.load(all_ckpt[0], map_location='cpu')
        ep, it = ckpt['epoch'], ckpt['iter']
        info.append(f'[auto_resume success] resume from ep{ep}, it{it}')
        return info, ep, it, ckpt['trainer'], ckpt['args']


def create_npz_from_sample_folder(sample_folder: str):
    """
    Builds a single .npz file from a folder of .png samples. Refer to DiT.
    """
    import os, glob
    import numpy as np
    from tqdm import tqdm
    from PIL import Image
    
    samples = []
    pngs = glob.glob(os.path.join(sample_folder, '*.png')) + glob.glob(os.path.join(sample_folder, '*.PNG'))
    assert len(pngs) == 50_000, f'{len(pngs)} png files found in {sample_folder}, but expected 50,000'
    for png in tqdm(pngs, desc='Building .npz file from samples (png only)'):
        with Image.open(png) as sample_pil:
            sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (50_000, samples.shape[1], samples.shape[2], 3)
    npz_path = f'{sample_folder}.npz'
    np.savez(npz_path, arr_0=samples)
    print(f'Saved .npz file to {npz_path} [shape={samples.shape}].')
    return npz_path

def save_img(path, image, file_name):
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, f"{file_name}.png")
    # if os.path.exists(save_path):
    #     return
    image.save(save_path)

def metrics(args, info):
    epoch = info["epoch"]
    iter = info["iter"]
    exp_name = info["exp_name"]

    testset_path = args.testset_path
    eval_img_num = args.eval_img_num
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(pyiqa.list_models())
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

    gt_img_paths = []

    psnr_folder = []
    ssim_folder = []
    lpips_score = []
    dists_score = []
    niqe_score = []
    lpips_iqa = []
    musiq_iqa = []
    maniqa_iqa = []
    clip_iqa = []
    gt_img_paths.extend(sorted(glob.glob(f'{testset_path}/gt/*.JPEG'))[:])
    gt_img_paths.extend(sorted(glob.glob(f'{testset_path}/gt/*.png'))[:])
    gt_img_folder = os.path.join(testset_path, "gt")
    if epoch is not None and iter is not None:
        generated_image_folder = os.path.join(testset_path, "VARPrediction", exp_name, f"{epoch}_{iter}")
    else:
        generated_image_folder = os.path.join(testset_path, "VARPrediction", exp_name)

    eval_img_num = len(gt_img_paths) if eval_img_num is None else min(eval_img_num, len(gt_img_paths))
    gt_img_paths = gt_img_paths[:eval_img_num]
    for gt_img_path in gt_img_paths:
        GT_image = img_preproc(Image.open(gt_img_path).convert('RGB'))
        prediction_img_path = os.path.join(generated_image_folder, os.path.basename(gt_img_path))
        VARPrediction_img = img_preproc(Image.open(prediction_img_path).convert('RGB'))

        img1 = rgb2ycbcr_pt(img2tensor(io.imread(gt_img_path)), y_only=True).to(torch.float64)
        img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)), y_only=True).to(torch.float64)
        img1 = torch.squeeze(img1)
        img2 = torch.squeeze(img2)

        ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
        psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
        lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_img_path))
        clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
        musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
        maniqa_iqa.append(maniqa_metric(prediction_img_path))
        dists_score.append(dists_iqa_metric(prediction_img_path, gt_img_path))
        niqe_score.append(niqe_iqa_metric(prediction_img_path))

    m_psnr = sum(psnr_folder) / len(psnr_folder)
    m_ssim = sum(ssim_folder) / len(ssim_folder)
    print(f"PSNR = {m_psnr}")
    print(f"SSIM = {m_ssim}")
    m_lpips = sum(lpips_iqa) / len(lpips_iqa)
    print(f"LPIPS = {m_lpips.item()}")
    m_dists = sum(dists_score) / len(dists_score)
    print(f"DISTS = {m_dists}")
    m_niqe = sum(niqe_score) / len(niqe_score)
    print(f"NIQE = {m_niqe}")
    clipiqa = sum(clip_iqa) / len(clip_iqa)
    print(f"CLIP-IQA = {clipiqa.item()}")
    musiq = sum(musiq_iqa) / len(musiq_iqa)
    print(f"MUSIQ = {musiq.item()}")
    maniqa = sum(maniqa_iqa) / len(maniqa_iqa)
    print(f"MANIQA = {maniqa.item()}")
    fid_value = fid_metric(gt_img_folder, generated_image_folder)
    print(f"FID = {fid_value}")

    with open(os.path.join(generated_image_folder, "metrics.txt"), "w") as f:
        f.write(f"eval_img_num: {eval_img_num}\n")
        f.write(f"PSNR: {m_psnr.item():.4f}\n")
        f.write(f"SSIM: {m_ssim.item():.4f}\n")
        f.write(f"LPIPS: {m_lpips.item():.4f}\n")
        f.write(f"DISTS: {m_dists.item():.4f}\n")
        f.write(f"NIQE: {m_niqe.item():.4f}\n")
        f.write(f"CLIP-IQA: {clipiqa.item():.4f}\n")
        f.write(f"MUSIQ: {musiq.item():.4f}\n")
        f.write(f"MANIQA: {maniqa.item():.4f}\n")
        f.write(f"FID: {fid_value.item():.4f}\n")

    return (m_psnr.item(), m_ssim.item(), m_lpips.item(), m_dists.item(), m_niqe.item(), clipiqa.item(), musiq.item(), maniqa.item(), fid_value.item())

def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy image.
    """
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    return images

def numpy_to_pil(images: np.ndarray):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor

def gaussian_weights(tile_width, tile_height, nbatches):
    """Generates a gaussian mask of weights for tile contributions"""
    from numpy import pi, exp, sqrt
    import numpy as np

    latent_width = tile_width
    latent_height = tile_height

    var = 0.01
    midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
    midpoint = latent_height / 2
    y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

    weights = np.outer(y_probs, x_probs)
    return torch.tile(torch.tensor(weights, device=dist.get_device()), (nbatches, 32, 1, 1))