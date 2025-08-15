from torchvision.transforms.transforms import F
from typing import Tuple, List, Union, Optional
from torch import device as Device, Tensor
from torchvision.utils import make_grid
from numpy import ndarray as Array
from itertools import product
from PIL import Image
from enum import Enum
from PIL import Image

import torch.nn as nn
import random as rand
import numpy as np
import cv2 as cv
import einops
import torch
import math
import calc


DEV_CPU = Device("cpu")
DEFAULT_DEVICE = DEV_CPU

class GaussianBlur(nn.Module):

    """torch 高斯滤波"""

    def __init__(
            self, 
            num_channels:int=1, 
            kernel_size:int=3, 
            sigma:float=0.5
        ) -> None:
        super().__init__()
        self.num_chan = num_channels
        self.kernel_size = kernel_size
        self.var = sigma ** 2
        xc = torch.arange(self.kernel_size).repeat(self.kernel_size)
        xc = einops.rearrange(xc, "(X Y) -> X Y", X=self.kernel_size)
        yc = einops.rearrange(xc, "X Y -> Y X", X=self.kernel_size)
        self.kernel_grid = torch.stack([xc, yc], dim=-1).to(torch.float32)
        mean = (self.kernel_size - 1) / 2.
        var = sigma ** 2
        ## 根据坐标阵构造高斯核
        self.kernel_pat = (1. / (2 * torch.pi * var) * torch.exp(-torch.sum((self.kernel_grid - mean) ** 2, dim=-1) / (2 * var)))
        self.kernel_pat /= torch.sum(self.kernel_pat)
        self.kernel_pat = einops.rearrange(self.kernel_pat, "X Y -> 1 1 X Y")
        self.kernel_pat = einops.repeat(self.kernel_pat, "C O X Y -> (C R) O X Y", R=self.num_chan)
        ## 将其封入卷积层并丢掉梯度
        self.kernel = torch.nn.Conv2d(self.num_chan, self.num_chan, self.kernel_size, groups=self.num_chan, bias=False, padding=self.kernel_size//2)
        self.kernel.weight.data = self.kernel_pat
        self.kernel.weight.requires_grad = False

    def to(self, *args, **kvs) -> None:
        super().to(*args, **kvs)
        self.kernel = self.kernel.to(*args, **kvs)
        return self

    def __call__(self, x:Tensor) -> Tensor: # for pyl
        return super().__call__(x)
        
    def forward(self, x:Tensor) -> Tensor:
        if self.num_chan != 1: return self.kernel(x)
        else: return einops.rearrange(self.kernel(x), "B 1 H W -> B H W")
    
class RGB2YCbCr(torch.nn.Module):

    """torch RGB <-> YCbCr 转换器 支持五维"""

    def __init__(
            self, 
            backward:bool=False, 
            bgr:bool=False, 
            ycrcb:bool=False, 
            device:Device=DEFAULT_DEVICE
        ) -> None:
        super().__init__()
        self.backward = backward
        self.mat = torch.tensor([[0.299, 0.587, 0.114], 
                                 [-0.169, -0.331, 0.5], 
                                 [0.5, -0.419, -0.081]], 
                                 dtype=torch.float32,
                                 device=device)
        if self.backward: self.inv = torch.inverse(self.mat).to(device)
        self.bias = torch.tensor([0., .5, .5], dtype=torch.float32, device=device)
        self.bgr = bgr
        self.ycrcb = ycrcb
        self.bias = einops.rearrange(self.bias, "C -> 1 C 1 1")

    def to(self, *args, **kvs) -> None:
        super().to(*args, **kvs)
        self.mat = self.mat.to(*args, **kvs)
        if self.backward: self.inv = self.inv.to(*args, **kvs)
        self.bias = self.bias.to(*args, **kvs)
        return self

    def __call__(self, x:Tensor) -> Tensor: # for pyl
        return super().__call__(x)

    def forward(self, x:Tensor) -> Tensor:
        assert x.shape[-3] == 3 and x.dtype == torch.float32
        fd_flag = x.ndim == 5
        if fd_flag: 
            GH = x.shape[0]
            x = einops.rearrange(x, "GH GW C H W -> (GH GW) C H W")
        elif x.ndim == 4: pass
        elif x.ndim == 3: x = x[None, ...]
        else: raise NotImplementedError
        if self.backward:
            if self.ycrcb: x = x[:, [2, 1, 0], ...] # YCbCr in
            x = (einops.einsum(self.inv, x - self.bias, "C D, B D H W -> B C H W")).clip(0., 1.)
            if self.bgr: x = x[:, [2, 1, 0], ...] # BGR out
        else:
            if self.bgr: x = x[:, [2, 1, 0], ...] # RGB in
            x = (einops.einsum(self.mat, x, "C D, B D H W -> B C H W") + self.bias).clip(0., 1.)
            if self.ycrcb: x = x[:, [2, 1, 0], ...] # YCrCb out
        if fd_flag: x = einops.rearrange(x, "(GH GW) C H W -> GH GW C H W", GH=GH)
        return x

def get_diff_colors(
        space_steps:int=3, 
        val_range:Tuple[int, int]=(0, 255), 
        channels:int=3, 
        shuffle:bool=False, 
        dtype:type=np.uint8
    ) -> Array:

    """获取 RGB 等距区分的颜色 主要用于分割结果展示"""

    aa = np.arange(val_range[0], val_range[1] + 1e-7, (val_range[1] - val_range[0]) / (space_steps - 1)).round().clip(*val_range).astype(dtype)
    diff_colors = [e for e in product(aa, repeat=channels)]
    if shuffle: rand.shuffle(diff_colors)
    return np.array(diff_colors, dtype=dtype)

def get_uint8_image_diff(
        image1:np.ndarray, 
        image2:np.ndarray, 
        visualize:bool=True
    ) -> np.ndarray:

    """获取 uint8 数组格式图像的差分 并指定放大差异"""

    diff = image1.astype(np.float32) - image2.astype(np.float32)
    diff = (diff / 2. + 127.5).clip(0, 255)
    if not visualize: return diff.astype(np.uint8)
    min, max = diff.min(), diff.max()
    diff = ((diff - min) / (max - min) * 255).clip(0, 255)
    return diff.astype(np.uint8)

def preprocess_file(path:str, device:Device=DEFAULT_DEVICE, resize=None):
    image = Image.open(path).convert("RGB")
    if resize is not None: image = image.resize(resize)
    image = np.array(image)
    return preprocess_np(image, device)

def preprocess_np(arr:Array, device:Device=DEFAULT_DEVICE):
    image = torch.from_numpy(arr).to(torch.float32)
    image = einops.rearrange(image / 127.5 - 1., "H W C -> C H W")
    return image.to(device)

def postprocess_2_np(ts:Tensor):
    if ts.ndim == 3: ts = ts[None, ...]
    ts = ((ts.detach().cpu() + 1.) * 127.5).round().clip(0, 255).to(torch.uint8)
    return einops.rearrange(ts, "B C H W -> B H W C").numpy()

def preprocess_sources(
        *sources:Union[str, Array, Tensor],
        resizes:Optional[Tuple[int]]=(224, 224),
        is_01:bool=True,
        device:Device=DEFAULT_DEVICE,
    ) -> Tensor:
    
    if isinstance(resizes, int): resizes = (resizes, resizes)
    collector = list()
    for src in sources:
        if isinstance(src, str):
            image = Image.open(src).convert("RGB")
            if resizes: image = image.resize(resizes, Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image)).to(torch.float32)
            image = (image / 255) if is_01 else (image / 127.5 - 1.)
            image = einops.rearrange(image, "H W C -> 1 C H W")
        elif isinstance(src, Array):
            if resizes: src = cv.resize(src, resizes, interpolation=cv.INTER_LINEAR)
            if src.dtype == np.uint8:
                image = torch.from_numpy(src).to(torch.float32)
                image = (image / 255) if is_01 else (image / 127.5 - 1.)
            elif src.dtype in [np.float32, np.float64]:
                image = torch.from_numpy(src).to(torch.float32)
                minn = image.min(); maxx = image.max()
                if minn >= 0. and maxx <= 1. and not is_01: image = image * 2 - 1
                elif minn >= -1. and maxx <= 1. and is_01: image = (image + 1.) / 2.
                elif minn >= 0. and maxx <= 255.:
                    image = (image / 255) if is_01 else image / 127.5 - 1.
                else: raise NotImplementedError
            else: raise NotImplementedError
            image = einops.rearrange(image, "H W C -> 1 C H W")
        elif isinstance(src, Tensor):
            image = src.clone().to(DEV_CPU)
            if image.ndim not in [3, 4]: raise NotImplementedError
            if image.ndim == 3: image = einops.rearrange(image, "C H W -> 1 C H W")
            if image.dtype in [torch.float32, torch.float64]:
                minn = image.min().item(); maxx = image.max().item()
                if minn >= 0. and maxx <= 1. and not is_01: image = image * 2 - 1
                elif minn >= -1. and maxx <= 1. and is_01: image = (image + 1.) / 2.
                elif minn >= 0. and maxx <= 255.:
                    image = (image / 255) if is_01 else image / 127.5 - 1.
                else: raise NotImplementedError
            elif src.dtype == torch.uint8:
                image = src.to(torch.float32)
                image = (image / 255) if is_01 else image / 127.5 - 1.
            else: raise NotImplementedError
            if resizes: image = F.resize(image, resizes, interpolation=F.InterpolationMode.BILINEAR)
        collector.append(image)
    batch = torch.cat(collector, dim=0).to(device)
    return batch

def monitor_channels(
        tensor:Tensor, 
        padding:int=2, 
        overall_redist:bool=False
    ) -> Array:

    """拼合张量的每个特征通道为大图 不支持批次"""

    if tensor.ndim == 4: tensor = tensor[0]
    assert tensor.ndim == 3
    C, H, W = tensor.shape
    row = int(math.sqrt(C)) # make sure row <= col
    col = math.ceil(C / row)
    imstack = torch.zeros((row, col, 1, H, W), dtype=torch.uint8).to(tensor.device)
    if overall_redist: tensor = calc.redist(tensor, mean=127.5, std=127.5, truncate=True)
    if overall_redist is None: 
        overall_redist = True
        tensor = calc.repave(tensor, min=0, max=255)
    for i in range(C):
        redisted = tensor[i]
        if not overall_redist:
            redisted = calc.redist(redisted, mean=127.5, std=127.5, truncate=True)
        redisted = torch.round(redisted).clamp(0, 255).to(torch.uint8)
        imstack[i // col, i % col, 0] = redisted

    grid = tile_grid(imstack, padding).cpu().numpy()
    return einops.rearrange(grid, "C H W -> H W C")

def make_tile(
        tensor:Tensor, 
        nh:int=0, 
        nw:int=0, 
        rear_comp:bool=True,
        device:Device=DEFAULT_DEVICE,
    ) -> Tensor:

    """根据图像张量 (B/N, C, H, W) 产生瓦片张量 (NH, NW, C, H, W)"""

    assert tensor.ndim == 4
    assert nw >= 0 and nh >= 0 
    B, C, H, W = tensor.shape

    if nh == nw == 0: 
        factor = math.sqrt(B)
        nw = math.ceil(factor)
        nh = math.ceil(B / nw)
    if nh == 0: nh = math.ceil(B / nw)
    if nw == 0: 
        nw = math.ceil(B / nh)

    if device is None: device = tensor.device
    tensor = tensor.to(device)

    diff = nh * nw - B
    assert diff >= 0

    if diff:
        comp = torch.zeros((diff, C, H, W)).to(device)
        cat_ls = [tensor, comp] if rear_comp else [comp, tensor]
        tensor = torch.cat(cat_ls, dim=0)
    
    # RH = nw if n_T else nh
    return einops.rearrange(tensor, "(RH RW) ... -> RH RW ...", RH=nh)

def make_tile_grid(
        tensor:Tensor, 
        nh:int=0, 
        nw:int=0, 
        rear_comp:bool=True, 
        padding:int=0, 
        device:Device=DEFAULT_DEVICE
    ) -> Tensor:

    """根据图像张量 (B, C, H, W) 经由瓦片张量 (NH, NW, C, H, W) 产生网格图"""

    tensor = make_tile(tensor, nh, nw, rear_comp, device)
    nrow = tensor.shape[1]
    tensor = einops.rearrange(tensor, "A B ... -> (A B) ...")
    return make_grid(tensor, nrow=nrow, padding=padding)

def tile_grid(
        tensor:Tensor, 
        padding:int=0
    ):

    """根据瓦片张量 (NH, NW, C, H, W) 产生网格图"""

    nrow = tensor.shape[1]
    tensor = einops.rearrange(tensor, "A B ... -> (A B) ...")
    return make_grid(tensor, nrow=nrow, padding=padding)

class TCDim(Enum):
    H = 0
    W = 1

DIM_H = TCDim.H
DIM_W = TCDim.W

def tile_cat(
        tensors:List[Tensor], 
        dim:TCDim=TCDim.H, 
        rear_comp:bool=True, 
        device:Device=DEFAULT_DEVICE
    ) -> Tensor:

    """对瓦片张量 (NH, NW, C, H, W) 进行拼接 自动补黑"""

    assert len(tensors) > 1
    for e in tensors[1:]: 
        assert len(tensors[0].shape) in [4, 5]
        assert len(e.shape) == len(tensors[0].shape)
        if len(e[0].shape) == 4: assert e.shape[-2:] == tensors[0].shape[-2:]
        else: assert e.shape[-3:] == tensors[0].shape[-3:]

    longest = 0
    for e in tensors: longest = max(longest, e.shape[dim.value])
    for e in tensors: 
        delta = longest - e.shape[dim.value]
        if delta == 0: continue
        if dim == TCDim.H: comp = torch.zeros([delta, *e.shape[1:]])
        else: comp = torch.zeros([e.shape[0], delta, *e.shape[2:]])
        comp = comp.to(device)
        if rear_comp: e = torch.cat([e, comp], dim=dim.value)
        else: e = torch.cat([comp, e], dim=dim.value)

    return torch.cat(tensors, dim=dim.value)


if __name__ == "__main__":

    import console as cs

    # dcs = get_diff_colors(3, shuffle=True)
    # print(dcs)
    # tile_test = torch.zeros((27, 1, 1, 1), dtype=torch.uint8)
    # count = 0
    # for i in range(27):
    #     tile_test[i, ...] = count
    #     count += 1
    # tg = make_tile_grid(tile_test, nh=5, rear_comp=False)
    # print(tg.squeeze())

    cs.pakprint(make_tile(torch.randn((10, 3, 32, 32)), nw=1))

    test_tl = [
        torch.randn((1, 10, 3, 112, 112)),
        torch.randn((1, 10, 3, 112, 112)),
        torch.randn((10, 10, 3, 112, 112)),
        torch.randn((1, 10, 3, 112, 112)),
        torch.randn((1, 10, 3, 112, 112)),
    ]

    tiled = tile_cat(test_tl, dim=TCDim.H)

    cs.pakprint(tiled)

