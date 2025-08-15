from typing import Union, List, Tuple, Callable, TypeVar, Generic
from functools import partial, reduce

import numpy as np
import einops
import torch
import copy


Number = Union[int, float, complex]
Tensor = torch.Tensor
Array = np.ndarray
Series = Union[List, Tuple, Array, Tensor]
Element = TypeVar("E")

EPS = 1e-9

def po(*elements:Element) -> Element:

    """连乘"""

    rst = reduce(lambda a, b: a * b, elements, 1.)
    return rst if abs(rst - round(rst)) > 1e-7 else round(rst)


def data_size(*arrs:Series) -> int:

    """张量格式元素个数"""

    def get_size(arr:Series) -> int:
        if isinstance(arr, Array): return arr.size
        elif isinstance(arr, Tensor): return arr.numel()
        else: return po(*arr)
    return round(sum([get_size(arr) for arr in arrs]))


def redist(
        oval:Element, 
        mean:float=0.5, 
        std:float=0.5, 
        truncate:bool=False
    ) -> Element:

    """重分布"""

    if isinstance(oval, Tensor):
        val = torch.clone(oval)
        omean = val.mean().item()
        ostd = val.std().item()
        val = ((val - omean) * (std / (ostd + EPS))) + mean
        if truncate: val = val.clamp(mean - std, mean + std)
    elif isinstance(oval, Array):
        val = np.copy(oval)
        omean = np.mean(val)
        ostd = np.std(val)
        val = ((val - omean) * (std / (ostd + EPS))) + mean
        if truncate: val = val.clip(mean - std, mean + std)
    else: raise NotImplementedError
    return val


def repave(
        oval:Element, 
        min:float=0., 
        max:float=1.
    ) -> Element:

    """重铺盖"""

    if isinstance(oval, torch.Tensor):
        val = torch.clone(oval)
        omin = val.min().item()
        omax = val.max().item()
        val *= (max - min) / (omax - omin + EPS)
        val += min - val.min().item()
        val = val.clamp(min, max)
    elif isinstance(oval, np.ndarray):
        val = np.copy(oval)
        omin = val.min()
        omax = val.max()
        val *= (max - min) / (omax - omin + EPS)
        val += min - val.min()
        val: np.ndarray
        val = val.clip(min, max)
    else: raise NotImplementedError
    return val


def zigzag(
        row_h:int, 
        col_w:int, 
        w_plus_first:bool=True, 
        ret_invert:bool=False
    ) -> List[int]:
    
    """二维 Z 字型取值 优先低座标 适配 DCT 压缩"""

    H, W = row_h, col_w
    seri = np.arange(H * W, dtype=np.int32)
    zz = np.zeros_like(seri, dtype=np.int32)
    idx = einops.rearrange(seri, "(H W) -> H W", H=H)

    def try_slope_ax(seq, x, y):
        nonlocal H, zz, idx
        while x - 1 >= 0 and y + 1 < H:
            x -= 1; y += 1; seq += 1
            zz[seq] = idx[y][x]
        return seq, x, y
    def try_slope_ay(seq, x, y):
        nonlocal W, zz, idx
        while y - 1 >= 0 and x + 1 < W:
            y -= 1; x += 1; seq += 1
            zz[seq] = idx[y][x]
        return seq, x, y

    def try_move_x(seq, x, y):
        nonlocal W, zz, idx
        if x + 1 < W:
            x += 1; seq += 1
            zz[seq] = idx[y][x]
        return seq, x, y
    def try_move_y(seq, x, y):
        nonlocal H, zz, idx
        if y + 1 < H:
            y += 1; seq += 1
            zz[seq] = idx[y][x]
        return seq, x, y
    
    def step_move():
        while True:
            yield try_move_x if w_plus_first else try_move_y
            yield try_move_y if w_plus_first else try_move_x

    def step_slope():
        while True:
            yield try_slope_ax if w_plus_first else try_slope_ay
            yield try_slope_ay if w_plus_first else try_slope_ax

    gen_move = step_move()
    gen_slope = step_slope()

    seq, x, y = 0, 0, 0
    while seq < len(zz) - 1:
        mseq, x, y = next(gen_move)(seq, x, y)
        if mseq == seq: 
            mseq, x, y = next(gen_move)(mseq, x, y)
            next(gen_move)
        seq = mseq
        seq, x, y = next(gen_slope)(seq, x, y)

    front = zz.tolist()
    if ret_invert:
        invert = list(sorted(front, key=lambda i: zz[i]))
        return front, invert
    return front


def check_l2_similarity(
        norms:Tensor, 
        features:Tensor, 
        norm_thresh:float=18.0, 
        similarity_threshs:List[float]=[0.3, 0.9]
    ) -> List[int]:

    """检查 L2 和相似度 返回被筛除的下标列表"""

    drop_indices = []

    ## 排除 L2 过小的
    except_norm = norms[:, 0] < norm_thresh
    drop_indices.extend(torch.nonzero(except_norm).squeeze(1).tolist())

    similarity_scores = features @ features.T

    ## 排除相似度过大的
    except_similarity = torch.nonzero(similarity_scores > similarity_threshs[1])
    upper_triangle = except_similarity[:, 0] < except_similarity[:, 1]
    drop_indices.extend(except_similarity[upper_triangle, 0].tolist())

    ## 排除平均相似度过小的
    row_ave_scores = (torch.sum(similarity_scores, dim=1) - 1.0) / (features.shape[0] - 1)
    drop_indices.extend(torch.nonzero(row_ave_scores < similarity_threshs[0]).squeeze().tolist())

    ## 去重作为列表返回
    return list(set(drop_indices))


R_FP = partial(round, ndigits=5)
R_NP = partial(np.round, decimals=5)

    
class MA(Generic[Element]):

    """滑动平均"""

    def __init__(
            self, 
            init:Union[Number, Array]=None, 
            i_num_eles:int=0, 
            need_copy:bool=True, 
            rounding:Callable[[Element], Element]=R_FP
        ) -> None:
        self.num_eles = i_num_eles
        self.need_copy = need_copy
        self.if_flag = None
        self.ma_rst = None
        self.func_round = rounding
        if init is not None: self.feed(init)

    def feed(self, element:Element) -> None:
        now_eles = self.num_eles + 1
        if self.if_flag is None:
            self.if_flag = isinstance(element, (int, float))
            self.func_round = R_FP if self.if_flag else R_NP
            self.ma_rst = float(element) if self.if_flag else self.copie(element)
            self.num_eles = now_eles
            return
        if self.num_eles == 0: self.ma_rst = element if self.if_flag else self.copie((element))
        self.ma_rst = self.ma_rst * (self.num_eles / now_eles) + element / now_eles
        self.num_eles = now_eles
    
    def copie(self, ele:Element) -> Element:
        return copy.deepcopy(ele) if self.need_copy else ele
    
    def __call__(self, element:Element) -> None: self.feed(element)

    def result(self, round=True) -> Element:
        rst = self.ma_rst if self.if_flag else self.copie(self.ma_rst)
        return self.func_round(rst) if round else rst
        

if __name__ == "__main__":

    # ls = list(range(1000000))
    import numpy as np
    np_ls = [np.ones((3, 64, 64)) for _ in range(1000)]

    import console as cs
    ma = MA(rounding=partial(np.round, decimals=5))

    with cs.TimeCost("av"):
        # summary = 0
        # for e in ls: summary += e
        # av = summary / len(ls)
        summary = np.copy(np_ls[0])
        for i, e in enumerate(np_ls[1:]): 
            summary += e
            av = summary / (i + 2)

    with cs.TimeCost("ma"):
        
        # for e in ls: ma.feed(e)
        for e in np_ls: ma(e)

    # print("ma: ", ma.result())
    # print("av: ", av)

    print("ma: ", ma.result()[0,0,0])
    print("av: ", av[0,0,0])

    print(zigzag(6, 7))

