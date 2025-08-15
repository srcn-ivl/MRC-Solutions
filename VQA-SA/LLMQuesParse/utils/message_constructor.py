from typing import List, Union
from PIL import Image

import imagesize as ims


class Content(list):

    def __init__(self):
        super().__init__()

    def add_text(self, text:str, **kvs):
        self.append({"type": "text", "text": text, **kvs})
        return self
    
    def add_image(self, image:Union[str, Image.Image], resize:tuple=None, **kvs):
        infos = {"type": "image", "image": image, **kvs}
        if resize: 
            if isinstance(resize, int):
                size = ims.get(image) if isinstance(image, str) else image.size
                ratio = resize / max(size)
                resize = (round(size[0] * ratio), round(size[1] * ratio))
            infos.update({"resized_height": resize[1], "resized_width": resize[0]})
        self.append(infos)
        return self
    
    def add_raw_images(self, image_paths:Union[str, List[str]]):
        if isinstance(image_paths, list):
            for ip in image_paths: self.append({"type": "image", "image": ip})     
        else: self.append({"type": "image", "image": image_paths})
        return self
    
    def add_video(self, mp4_path:str, max_pixels:int=None, fps:float=1.0, **kvs):
        block = {"type": "video", "video": mp4_path, "fps": fps, **kvs}
        if max_pixels: block["max_pixels"] = max_pixels
        self.append(block)
        return self
 
    def add_frames(self, frames:List[str], **kvs):
        self.append({"type": "video", "video": frames, **kvs})
        return self
    
    def __repr__(self):
        import json
        return json.dumps(self, indent=4)


class Message(list):

    def __init__(self):
        super().__init__()

    def add_role_block(self, role:str, content:list):
        self.append({"role": role, "content": content})
        return self
    
    def copy(self):
        import copy
        return copy.deepcopy(self)
    
    def __repr__(self):
        import json
        return json.dumps(self, indent=4, default=lambda o: repr(o), ensure_ascii=False)
    

class Box:
    xs:int
    ys:int
    xe:int
    ye:int
    _normed:bool = False

    def norm(self, W:int=1920, H:int=1080):
        if self._normed: return self
        self.xs = round(self.xs * W / 1000)
        self.xe = round(self.xe * W / 1000)
        self.ys = round(self.ys * H / 1000)
        self.ye = round(self.ye * H / 1000)
        self._normed = True
        return self
    
    def __repr__(self):
        return f"({self.xs}, {self.ys}), ({self.xe}, {self.ye})"
    
    def point(self, end:bool=False):
        return (self.xe, self.ye) if end else (self.xs, self.ys) 

