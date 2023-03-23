"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

import os
import sys
import cv2
import time
import math
import h5py
import copy
import torch
import random
import argparse
import datetime
import warnings
import colorsys
import requests
import subprocess
import skimage.io
import torchvision
import numpy as np
from torch import nn
from PIL import Image
from io import BytesIO
import random_resized_crop
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from PIL import ImageFilter, ImageOps
from matplotlib.patches import Polygon
from tensorboardX import SummaryWriter
from skimage.measure import find_contours
from collections import defaultdict, deque
from torchvision import datasets, transforms

from spectral_cluster import *
from fseval import test_methods


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class MultiViewTransform(object):
    
    def __init__(
        self,
        rand_transform=None,
        focal_transform=None,
        rand_views=1,
        focal_views=1,
    ):
        self.rand_views = rand_views
        self.focal_views = focal_views
        self.rand_transform = rand_transform
        self.focal_transform = focal_transform

    def __call__(self, img):
        img_views = []

        # -- generate random views
        if self.rand_views > 0:
            img_views += [self.rand_transform(img) for i in range(self.rand_views)]

        # -- generate focal views
        if self.focal_views > 0:
            img_views += [self.focal_transform(img) for i in range(self.focal_views)]

        return img_views

def make_transforms(
    rand_size=224,
    focal_size=96,
    rand_crop_scale=(0.3, 1.0),
    focal_crop_scale=(0.05, 0.3),
    color_jitter=1.0,
    rand_views=2,
    focal_views=10,
):

    def get_color_distortion(s=1.0):
        # s is the strength of color distortion.
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort

    rand_transform = transforms.Compose([
        transforms.RandomResizedCrop(rand_size, scale=rand_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])
    focal_transform = transforms.Compose([
        transforms.RandomResizedCrop(focal_size, scale=focal_crop_scale),
        transforms.RandomHorizontalFlip(),
        get_color_distortion(s=color_jitter),
        GaussianBlur(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225))
    ])

    transform = MultiViewTransform(
        rand_transform=rand_transform,
        focal_transform=focal_transform,
        rand_views=rand_views,
        focal_views=focal_views
    )
    return transform

def load_pretrained_weights(model, pretrained_weights, checkpoint_key, model_name=None, patch_size=None):
    if os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu")
        if checkpoint_key is not None and checkpoint_key in state_dict:
            print(f"Take key {checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(pretrained_weights, msg))
    else:
        print("There is no reference weights available for this model => We use random weights.")

def clip_gradients(model, clip):
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms

def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
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
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(torch.tensor(input_dict[k], device='cuda'))
        
        values = torch.stack(values, dim=0)

        dist.reduce(values,0) # reduce to all processes
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict

class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
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
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.world_size = 0, 1
        if args.gpu == None:
          args.gpu =0 
       
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    if args.rank == 0:
        set_writer(args.output_dir)
        print(f"\nrank{args.rank} set tb writer\n")

        from pathlib import Path
        import json
        # Save args
        with (Path(args.output_dir) / "args.txt").open("a") as f:
            f.write("args:\n")
            f.write(json.dumps(dict(vars(args))) + "\n\n")

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            print("_out.shape", _out.shape)
            print("output.shape", output.shape)
            
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class MultiCropWrapper_smkd(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper_smkd, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        if head is None:
            self.head = nn.Identity()
        else:
            self.head = head

    def forward(self, x, mask=None, return_backbone_feat=False, return_attention=False, **kwargs):
        # convert to list
        if not isinstance(x, list):
            x = [x]
            mask = [mask] if mask is not None else None
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0) 
        start_idx = 0
        for end_idx in idx_crops:
            inp_x = torch.cat(x[start_idx: end_idx])

            if mask is not None:
                inp_m = torch.cat(mask[start_idx: end_idx])
                kwargs.update(dict(mask=inp_m))

            _out, attn = self.backbone(inp_x, return_attention=return_attention, **kwargs)
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        output_ = self.head(output)
        if return_backbone_feat:
            return output, output_, attn
        return output_, attn

def token_pooling(attn_maps, patch_tokens, N, labels=None):
    labels = spectral_cluster(attn_maps, K=int(N/2), pre_labels=labels)
    patch_tokens = cluster_reduce(patch_tokens, labels, int(N/2))
    return patch_tokens.cuda(),labels.cuda()

def cluster_reduce(feats,labels,K):
    B,_,D = feats.shape

    result = torch.zeros(B,K,384)

    for ind,feat in enumerate(feats):
        label = labels[ind] # (784,)
        label = label.view(label.size(0), 1).expand(-1, feat.size(1))

        classes_,counts_ = torch.unique(label,sorted=True,return_counts=True, dim=0)

        result[ind] = torch.zeros_like(classes_, dtype=torch.float).scatter_add_(0, label.long(), feat).cuda()
        result[ind] = result[ind].cuda() / counts_.float().unsqueeze(1).cuda()

    return result

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def multi_scale(samples, model):
    v = None
    for s in [1, 1/2**(1/2), 1/2]:  # we use 3 different scales
        if s == 1:
            inp = samples.clone()
        else:
            inp = nn.functional.interpolate(samples, scale_factor=s, mode='bilinear', align_corners=False)
        feats = model(inp).clone()
        if v is None:
            v = feats
        else:
            v += feats
    v /= 3
    v /= v.norm()
    return v

writer = None

def set_writer(output_dir):
    from pathlib import Path

    global writer
    path = os.path.join(output_dir,'tb/')
    Path(path).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(path)
    
def save_tb(log_stats):
    if writer is None:
        return
    epoch = log_stats['epoch']
    del log_stats['epoch']

    for k,v in log_stats.items():
        writer.add_scalar(k,v,epoch)

def save_tb_iter(log_stats_, it_global):
    reduced_stat = reduce_dict(log_stats_)

    if writer is None:
        return
    for k,v in reduced_stat.items():
        writer.add_scalar(f'train_{k}_it',v, it_global)

def save_command(file_path):
    args = copy.deepcopy(sys.argv)
    args[0] = os.path.basename(args[0])
    command = ' '.join(['python'] + args)
    print(command)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    text_file = open(file_path, "w")
    text_file.write(command)
    text_file.close()
    
class DataAugmentationSMKD(object):
    def __init__(self, global_crops_scale, local_crops_scale, global_crops_number, local_crops_number, 
                 global_crops_size, local_crops_size, dataset_name):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        '''
        mini/tiered:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        cifar/FC100:
            mean = [x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]]
            std = [x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]]

        '''
        
        if 'mini' in dataset_name or 'tiered' in dataset_name:
            mean = tuple([0.485, 0.456, 0.406])
            std = tuple([0.229, 0.224, 0.225])
        elif 'cifar' in dataset_name or 'FC100' in dataset_name:
            mean = tuple([x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]])
            std = tuple([x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]])  
       
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
            
        self.global_crops_number = global_crops_number
        # transformation for the first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # transformation for the rest of global crops
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        for _ in range(self.global_crops_number - 1):
            crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

########### Evaluation ############

def evaluation(args, vits, epoch, rollout=False, foreground_threshold = 0.4, evaluation_method='cosine', top_k=5, 
               img_size_imagenet=256, img_resize_imagenet=224, img_size_cifar=256, img_resize_cifar=224):

    evaluate_data_path = os.path.dirname(args.data_path)
    evaluate_ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')

    #server = server_dict[args.server]
    print("git:\n  {}\n".format(get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    dataset_name = args.dataset_name 
    
    if 'mini' in dataset_name or 'tiered' in dataset_name:
        mean = tuple([0.485, 0.456, 0.406])
        std = tuple([0.229, 0.224, 0.225])
        img_size = img_size_imagenet 
        img_resize = img_resize_imagenet 
    elif 'cifar' in dataset_name or 'FC100' in dataset_name:
        mean = tuple([x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]])
        std = tuple([x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]])  
        img_size = img_size_cifar 
        img_resize = img_resize_cifar    
    # ============ preparing data ... ============
    val_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=3),
        transforms.CenterCrop(img_resize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset_test = datasets.ImageFolder(os.path.join(evaluate_data_path, args.partition), transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.evaluation_batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_test)} test imgs.")

    # ============ building network ... ============
    import models as mymodels
    if args.arch in mymodels.__dict__.keys():
        model = mymodels.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    
    print(f"Evaluating pretrained weight in {evaluate_ckpt_path}")
    
    args.evaluate_output_dir =  os.path.dirname(evaluate_ckpt_path)

    server = {'dataset': 'mini',
              'data_path': evaluate_data_path,   # Need to modify here
              'ckp_path': evaluate_ckpt_path}

    outfile = os.path.join(args.evaluate_output_dir,'{}_224_epoch{}_{}_{}shots.hdf5'.format(args.partition,epoch, args.checkpoint_key, args.num_shots))
    if not os.path.isfile(outfile):
        load_pretrained_weights(model, evaluate_ckpt_path, args.checkpoint_key, args.arch,
                                        args.patch_size)
        if args.save_features == 1:
            save_features(model,server['dataset'], test_loader, 1, 
                          args.cls_tokens, args.avgpool_patchtokens, args.weighted_avgpool_patchtokens, 
                          args.p_scale, args.wp_scale, 
                          epoch, server['ckp_path'],outfile)
    test_methods(args,server,epoch,os.path.dirname(server['ckp_path']),outfile, evaluation_method=evaluation_method, top_k=top_k)
        
    if int(args.epochs) == -1:
        return

def save_features(model,dataset,loader, n, clstokens, avgpool, weightedavgpool, p_scale, wp_scale, epochs, pretrained_weights,outfile):
    
    return_attention = True if weightedavgpool else False 
    
    print('outputfile:',outfile)
    
    f = h5py.File(outfile, 'w')
    max_count = len(loader) * loader.batch_size
    print(max_count)
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None

    count = 0
    for i, (inp, target) in enumerate(loader):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        # forward
        with torch.no_grad():
            intermediate_output, attns = model.get_intermediate_layers(inp, n, return_attention=return_attention)
            if clstokens:
                output = [x[:,0] for x in intermediate_output]
            else:
                output = []
            if avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1) * p_scale) 
            if attns is not None:
                if weightedavgpool:
                    patch_tokens = intermediate_output[-1][:, 1:] #[100, 196, 384]
                    attn_weights = attns[-1].mean(axis = 1)[:, 0, 1:] # [100, 196]
                    attn_weights = attn_weights / attn_weights.sum(axis = -1, keepdims = True)
                    weighted_patch_tokens = patch_tokens * attn_weights.unsqueeze(-1)
                    output.append(torch.sum(weighted_patch_tokens, dim=1) * wp_scale) 
            
            output = [nn.functional.normalize(x, dim=-1) for x in output]
            output = torch.cat(output, dim=-1)

        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(loader)))
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(output.size()[1:]), dtype='f')

        all_feats[count:count + output.size(0)] = output.data.cpu().numpy()
        all_labels[count:count + output.size(0)] = target.cpu().numpy()

        count = count + output.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()
    print(outfile)
    
######################### visualization #########################

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image

def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def visualize(args, vits, epoch, image_path = None, image_size = (480, 480), threshold = None, rollout=False):

    args.image_path = image_path
    args.image_size = image_size
    args.image_threshold = threshold
    args.visualize_output_dir = os.path.join(args.output_dir, "visualization_epoch{}".format(epoch))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    visualize_ckpt_path = os.path.join(args.output_dir, 'checkpoint.pth')

    if os.path.isfile(visualize_ckpt_path):
        state_dict = torch.load(visualize_ckpt_path, map_location="cpu")
        if args.checkpoint_key is not None and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
        # remove `module.` prefix
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        # remove `backbone.` prefix induced by multicrop wrapper
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(visualize_ckpt_path, msg))
    
    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
        
    dataset_name = args.dataset_name
    if 'mini' in dataset_name or 'tiered' in dataset_name:
        mean = tuple([0.485, 0.456, 0.406])
        std = tuple([0.229, 0.224, 0.225])
    elif 'cifar' in dataset_name or 'FC100' in dataset_name:
        mean = tuple([x/255.0 for x in [129.37731888,  124.10583864, 112.47758569]])
        std = tuple([x/255.0 for x in [68.20947949,  65.43124043,  70.45866994]])  
   
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)
    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device), rollout)
    attentions = attentions[1]
    nh = attentions.shape[1] # number of head
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.image_threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.image_threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.visualize_output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.visualize_output_dir, "img.png"))
    for j in range(nh):
        fname = os.path.join(args.visualize_output_dir, "attn-head" + str(j) + ".png")
        plt.imsave(fname=fname, arr=attentions[j], format='png')
        print(f"{fname} saved.")

    try:
        fig, ax = plt.subplots(2, 3, figsize = (7.2, 5))
        plt.subplots_adjust(wspace = 0, hspace = 0)
        i = 0
        for idx in [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)]:
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].imshow(attentions[i])
            i += 1
        #plt.show()
        fig.savefig(os.path.join(args.visualize_output_dir, "attn-heads" + ".png"), bbox_inches='tight', dpi = 500)
    except:
        fig, ax = plt.subplots(1, 3, figsize = (3.5, 5))
        plt.subplots_adjust(wspace = 0, hspace = 0)
        i = 0
        for idx in [(0,0), (0,1), (0,2)]:
            ax[idx].set_xticks([])
            ax[idx].set_yticks([])
            ax[idx].imshow(attentions[i])
            i += 1
        #plt.show()
        fig.savefig(os.path.join(args.visualize_output_dir, "attn-heads" + ".png"), bbox_inches='tight', dpi = 500)
    
    fig, ax = plt.subplots(1, 1)
    plt.imshow(attentions.mean(axis = 0))
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.show()
    fig.savefig(os.path.join(args.visualize_output_dir, "attn-combined" + ".png"), bbox_inches='tight', dpi = 500)
    
    if args.image_threshold is not None:
        image = skimage.io.imread(os.path.join(args.visualize_output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.visualizoutput_dir, "mask_th" + str(args.image_threshold) + "_head" + str(j) +".png"), blur=False)

def list_duplicates(seq, pseudo=False):
    tally = defaultdict(list)
    if pseudo:
        for i, item in enumerate(seq):
            tally[i].append(i)
    else:
        for i,item in enumerate(seq):
            tally[item].append(i)
    return ((key,locs) for key,locs in tally.items() if len(locs)>=1)
    
def crops_in_same_class(label_idx, batch_size_per_gpu, crops_number):
    result = []
    for c in range(crops_number):
        result += [idx + c*batch_size_per_gpu for idx in label_idx]
    return result 

def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



