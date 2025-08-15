from IPython.display import display, HTML
from typing import Union, Tuple, Dict
from PIL import Image

import numpy as np
import cv2 as cv
import einops
import torch
import json
import osp
import re


INTER_LINEAR = cv.INTER_LINEAR
INTER_LANCZOS = cv.INTER_LANCZOS4
INTER_NEAREST = cv.INTER_NEAREST

PILImage = Image.Image
Array = np.ndarray
Tensor = torch.Tensor

image_base64_pattern = r"image/(?:jpeg|png|gif|x\-icon|svg\+xml|bmp)"

def in_notebook():
    import sys
    return "ipykernel" in sys.modules

def display_image_files(*images:Union[str, PILImage], title=False) -> None:

    filenames = [img if isinstance(img, str) else img.filename for img in images]
    wraps = [f"<h3>{fn}</h3><img src=\"{fn}\"/>" if title else f"<img src=\"{fn}\"/>" for fn in filenames]
    html = "".join(wraps)
    display(HTML(html))

def display_pil_images(
        *images:PILImage, 
        interval:int=2
    ) -> None:

    images = [np.array(image) for image in images]
    display_np_images(*images, interval=interval)

def display_np_images(
        *images:Array, 
        interval:int=2, 
        resize:Union[int, float, Tuple[int], None]=None, 
        interpolation:int=INTER_LINEAR,
        bgr2rgb:bool=False
    ) -> None:

    if resize:
        if isinstance(resize, (int, float)): resize = [round(es * resize) for es in images[0].shape[:2]][::-1]
        images = [cv.resize(image, resize, interpolation=interpolation) for image in images]
    shape = list(images[0].shape)
    shape[1] = interval
    merged = [images[0]]
    for i in range(len(images) - 1): merged.extend([np.zeros(shape, dtype=np.uint8), images[i + 1]])
    merged = np.concatenate(merged, axis=1)
    if bgr2rgb: merged = cv.cvtColor(merged, cv.COLOR_BGR2RGB)
    display(Image.fromarray(merged))

def display_ts_images(
        *images:Tensor, 
        interval:int=2, 
        resize:Union[int, float, Tuple[int], None]=None, 
        interpolation:int=INTER_LINEAR
    ) -> None:

    images:Tensor = torch.cat(images).detach().cpu()
    maxx = images.max(); minn = images.min()
    ext_range = (maxx - minn).item()
    if 0 <= ext_range <= 2 and -1 <= minn < 0 and maxx <= 1: images = (images + 1.) * 127.5
    elif 0 <= ext_range <= 1 and minn >= 0: images = images * 255.
    elif 0 <= ext_range < 256 and minn >= 0: pass
    else: raise NotImplementedError
    images = einops.rearrange(images, "B C H W -> B H W C")
    images = images.round().clip(0., 255.).to(torch.uint8).numpy()
    display_np_images(*images, interval=interval, resize=resize, interpolation=interpolation)

def rm_ipynb_base64(
        path:str, 
        force:bool=False
    ) -> None:
    
    """
    消除 .ipynb 中以 base64 存在的图片  
    主要用于减少远程仓库占用  
    参数：  
    &emsp;path (str): 文件路径  
    &emsp;force (bool): 允许非 .ipynb 后缀文件  
    返回：None
    """

    assert osp.exists(path) and osp.isfile(path), "Path is not exist or a directory"
    if not path.endswith(".ipynb"):
        if not force: return
        import console as cs
        cs.cprint(f"Trying to process non ipynb suffix file: {osp.op_base(path)}", fore_color=cs.F_WARN)

    with open(path, "r", encoding="utf8") as reader:
        ipy_contents:dict = json.load(reader)

    for cell in ipy_contents.get("cells", list()):
        cell:dict
        if cell.get("cell_type", "null") != "code": continue
        if "outputs" not in cell: continue
        if not isinstance(cell["outputs"], list): continue
        for output in cell["outputs"]:
            if "data" not in output: continue
            if not isinstance(output["data"], dict): continue
            output:Dict[str, dict]
            keys = [k for k in output["data"].keys() if re.match(image_base64_pattern, k)]
            for key in keys: del output["data"][key]

    with open(path, "w", encoding="utf8") as writer:
        json.dump(ipy_contents, writer, indent=4)
            
