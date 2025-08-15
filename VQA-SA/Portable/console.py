from typing import List, Union, Any, Callable, Optional

from functools import partial
from enum import Enum

import time


class ForeColor(Enum):
    text = 37
    info = 34
    warn = 33
    error = 31
    done = 32
    unwill = 35


class BackColor(Enum):
    black = 40
    red = 41
    green = 42
    yellow = 43
    blue = 44
    purple = 45
    cyan = 46
    white = 47


class Effect(Enum):
    none = 0
    highlight = 1
    dark = 2
    italic = 3
    underline = 4
    glitter = 5
    inverse = 7
    invisible = 8


F_TEXT = ForeColor.text
F_INFO = ForeColor.info
F_WARN = ForeColor.warn
F_ERROR = ForeColor.error
F_DONE = ForeColor.done
F_UNWILL = ForeColor.unwill

B_BLACK = BackColor.black
B_RED = BackColor.red
B_GREEN = BackColor.green
B_YELLOW = BackColor.yellow
B_BLUE = BackColor.blue
B_PURPLE = BackColor.purple
B_CYAN = BackColor.cyan
B_WHITE = BackColor.white

E_NONE = Effect.none
E_HIGHLIGHT = Effect.highlight
E_DARK = Effect.dark
E_ITALIC = Effect.italic
E_UNDERLINE = Effect.underline
E_GLITTER = Effect.glitter
E_INVERSE = Effect.inverse
E_INVISIBLE = Effect.invisible


TAS_BASE = ["shape", "dtype"]
TAS_DIST = ["max", "min", "mean", "std"]
TAS_DEV = ["device"]
TAS_TYPE = ["type"]
TAS_BASEDEV = TAS_BASE + TAS_DEV
TAS_GRAD = ["requires_grad"]
TAS_TSBASE = TAS_BASEDEV + TAS_GRAD


COLOR_PRINT = True

def global_cprint(flag:bool=False) -> None:
    global COLOR_PRINT
    COLOR_PRINT = flag


def cprint(
        *content:str, 
        fore_color:ForeColor=F_INFO, 
        back_color:BackColor=None, 
        effect:Effect=None, 
        sep:str=" ", 
        end:str="\n"
    ) -> None:

    """控制台输出颜色与效果"""

    content = [ctext(c, fore_color, back_color, effect) for c in content]
    print(*content, sep=sep, end=end)    


def ctext(
        content:str, 
        fore_color:ForeColor=F_INFO, 
        back_color:BackColor=None, 
        effect:Effect=None
    ) -> str:

    """颜色与效果字符串"""

    global COLOR_PRINT
    command = ";".join([str(e.value) for e in [effect, fore_color, back_color] if e is not None])
    return f"\033[{command}m{content}\033[0m" if COLOR_PRINT else content


def get_ts_attr(
        tensor:Any,
        attr:List[str], 
        default:str="Not Found",
        rounding:int=None
    ) -> str:
    """attr of [shape, dtype, device, type, max, min, mean, std, var, ...]"""
    if attr == "shape": 
        at = getattr(tensor, "shape", default)
        if not isinstance(at, str): at = tuple(at)
    elif attr == "type": at = str(type(tensor))[8:-2]
    elif attr in ("max", "min", "mean", "std", "var"):
        at = getattr(tensor, attr, default)
        if not isinstance(at, str): at = at().item()
        if rounding is not None: at = round(at, rounding)
    else: at = getattr(tensor, attr, default)
    return str(at)


def dbgprint(
        *content:str,
        sep:str=" ", 
        sp:str="\n\n"
    ) -> None:

    """调试打印"""

    cprint(sp, fore_color=F_WARN)
    cprint(*content, fore_color=F_WARN, sep=sep)
    cprint(sp, fore_color=F_WARN)


def tstext(
        loc:dict, 
        *ts_names:str, 
        attrs:List[str]=TAS_BASE, 
        rounding:int=None
    ) -> str:

    """张量信息"""

    merged = f"{'Attributes':^21s}: {'| '.join([attr.center(16) for attr in attrs])}\n"

    for i, n in enumerate(ts_names):
        if not isinstance(n, str) or loc.get(n) is None:
            merged += f"{(i + 1):02d} - {n:16s}: Not Found\n"
            continue
        str_attrs = ", ".join([f"{get_ts_attr(loc.get(n), attr, rounding=rounding):^16s}" for attr in attrs])
        merged += f"{(i + 1):02d} - {n:^16s}: {str_attrs}\n"

    return merged


def itstext(
        *ts_names:str,
        attrs:List[str]=TAS_BASEDEV,
        rounding:int=5   
    ) -> str:

    """通过栈帧获取变量表和键 获取张量信息"""

    import inspect as ins
    outer_locals = ins.currentframe().f_back.f_locals
    return tstext(outer_locals, *ts_names, attrs=attrs, rounding=rounding)


def tsprint(
        loc:dict, 
        *ts_names:str, 
        attrs:List[str]=TAS_BASE, 
        color:bool=COLOR_PRINT,
        rounding:int=None
    ) -> None:

    """打印张量信息"""

    hprint = partial(cprint, fore_color=F_WARN) if color else print
    sprint = partial(cprint, fore_color=F_INFO) if color else print
    eprint = partial(cprint, fore_color=F_ERROR) if color else print

    hprint(f"{'Attributes':^21s}: {'| '.join([attr.center(16) for attr in attrs])}")

    for i, n in enumerate(ts_names):
        if not isinstance(n, str) or loc.get(n) is None:
            eprint(f"{(i + 1):02d} - {n:16s}: Not Found")
            continue
        str_attrs = ", ".join([f"{get_ts_attr(loc.get(n), attr, rounding=rounding):^16s}" for attr in attrs])
        hprint(f"{(i + 1):02d} - {n:^16s}", end="")
        sprint(f": {str_attrs}")


def itsprint(
        *ts_names:str,
        attrs:List[str]=TAS_BASEDEV,
        color:bool=COLOR_PRINT,
        rounding:int=5   
    ) -> None:

    """通过栈帧获取变量表和键 打印张量信息"""

    import inspect as ins
    outer_locals = ins.currentframe().f_back.f_locals
    tsprint(outer_locals, *ts_names, attrs=attrs, color=color, rounding=rounding)


def parse_dat_pak(ref:Any) -> Union[int, float, complex, str, bool, list, tuple, set, dict]:

    """解析数据包"""

    import numpy as np
    import torch

    type_map = {np.ndarray: "Array", torch.Tensor: "Tensor"}

    if isinstance(ref, dict):
        return {k:parse_dat_pak(v) for k, v in ref.items()}
    elif isinstance(ref, (int, float, complex, str, bool)):
        return ref
    elif isinstance(ref, (np.ndarray, torch.Tensor)):
        return f"{type_map.get(type(ref))}{get_ts_attr(ref, 'shape')}[{get_ts_attr(ref, 'dtype')}]"
    elif isinstance(ref, (list, tuple, set)):
        return type(ref)([parse_dat_pak(e) for e in ref])
    else: return f"{get_ts_attr(ref, 'type')}({str(ref)})"


def pakprint(
        pak:Any, 
        name:str="pak"
    ) -> None:

    """打印数据包格式"""

    import json

    print(f"{name}:", json.dumps(parse_dat_pak(pak), ensure_ascii=False, indent=4))


def get_time_str(format:str=r"%Y%m%d_%H%M%S") -> str:
    """获取时间串"""
    from datetime import datetime
    return datetime.now().strftime(format)


def deprecated(
        reason:str="", 
        color:bool=COLOR_PRINT, 
        cn:bool=True
    ):
    
    """弃用标记装饰器"""

    f_print = partial(cprint, fore_color=F_WARN) if color else print
    def to_func(func):
        def params(*args, **kwargs):
            if cn: f_print("-----弃用-----")
            else: f_print("--deprecated--")
            rst = func(*args, **kwargs)
            if cn: f_print(f"\n调用弃用函数 <{func.__name__}>\n{reason}")
            else: f_print(f"\ninvoked deprecated func <{func.__name__}>\n{reason}")
            f_print("--------------")
            return rst
        return params
    return to_func


class TimeUnit(Enum):
    ms = 0, 1000, 4
    s = 1, 1, 4
    us = 2, 1000000, 0


def time_counter(
        unit:TimeUnit=TimeUnit.ms, 
        color:bool=COLOR_PRINT, 
        counter:Callable[[], float]=time.perf_counter, 
        cn:bool=True
    ):

    """计时装饰器"""

    def to_func(func):
        def params(*args, **kwargs):
            start = counter()
            rst = func(*args, **kwargs)
            delta = (counter() - start) * unit.value[1]
            delta = round(delta, unit.value[2])
            if unit.value[2] == 0: delta = int(delta)
            if cn: prstr = f"函数 <{func.__name__}> 执行耗时：{delta}{unit.name}"
            else: prstr = f"Time of execute {func.__name__}:  {delta}{unit.name}"
            f_print = partial(cprint, fore_color=F_DONE) if color else print
            f_print(prstr)
            return rst
        return params
    return to_func


class TimeCost():

    """计时上下文"""

    def __init__(
            self, 
            name:Optional[str]=None, 
            unit:TimeUnit=TimeUnit.ms, 
            color:bool=COLOR_PRINT, 
            counter:Callable[[], float]=time.perf_counter, 
            cn:bool=False,
            ser:bool=False
        ) -> None:
        self.moment = 0
        self.name = "undefined" if name is None else name
        self.unit = unit
        self.f_print = partial(cprint, fore_color=F_DONE) if color else print
        self.counter = counter
        self.cn = cn
        self.ser = ser

    def __enter__(self):
        if not self.ser: self.moment = self.counter()
        return self
    
    def __exit__(self, *_) -> None:
        if self.ser: return
        delta = (self.counter() - self.moment) * self.unit.value[1]
        delta = round(delta, self.unit.value[2])
        if self.cn: prstr = f"代码段 <{self.name}> 执行耗时：{delta}{self.unit.name}"
        else: prstr = f"Time of execute code section <{self.name}>:  {delta}{self.unit.name}"
        self.f_print(prstr)


class StreamTimeCounter():

    """流程计时器 可返回 str 用于兼容多进程 logging"""

    def __init__(
            self, 
            unit:TimeUnit=TimeUnit.ms, 
            round:int=4, 
            counter:Callable[[], float]=time.perf_counter, 
            ret_str:bool=True
        ) -> None:
        self.unit = unit
        self.round = round
        self.counter = counter
        self.ret_str = ret_str
        self.start = self.counter()
        self.flow = self.start
        
    def sec_elapse(self):
        now = self.counter()
        delta = (now - self.flow) * self.unit.value[1]
        self.flow = now
        return self._result(delta)

    def overall_elapse(self):
        delta = (self.counter() - self.start) * self.unit.value[1]
        return self._result(delta)
    
    def _result(self, delta:float) -> Union[str, float, int]:
        delta = round(delta, self.round)
        return f"{delta}{self.unit.name}" if self.ret_str else delta


class IArgparse:

    """
    控制台参数分析上下文  
    通过调用栈获取调用区上下文内变量 以变量定义的类型和值分析为默认
    """

    def __init__(self, context_global:dict=dict()):
        self.callables = list()
        self.context:dict = context_global

    def __enter__(self):
        import inspect as ins
        self.record = {k: v for k, v in ins.currentframe().f_back.f_locals.items()}
        return self

    def parse_name(self, name:str):
        return name.strip("_").replace("_", "-")

    def parse_args(self, diff_locals:dict):
        import argparse
        parser = argparse.ArgumentParser()
        for name, default in diff_locals.items():
            arg_name = f"--{self.parse_name(name)}"
            arg_type = type(default)
            adds = dict()
            ## argparse 原生支持的基本参数类型
            if arg_type in [int, float, str, bool]: pass
            ## 多个参数扩展至列表和元组
            elif arg_type in [list, tuple]: 
                if len(default) == 0 : continue
                adds["nargs"] = "*"
                arg_type = type(default[0])
                if arg_type not in [int, float, str, bool]: continue
            ## 扩展参数范围至可调用对象
            elif callable(default): 
                self.callables.append(name)
                arg_type = str
            ## 其它类型不支持 不加入参数
            else: continue
            parser.add_argument(arg_name, type=arg_type, default=default, **adds)
        return parser
    
    def parse_callables(self, kvargs:dict) -> dict:
        for name in self.callables:
            if callable(kvargs[name]): continue
            func_sign:str = kvargs[name]
            ## 如果文件 globals 中有调用名
            if func_sign in self.context: kvargs[name] = self.context[func_sign]; continue
            invoke_chain = func_sign.split(".")
            ## 如果直接调用 在 builtins 中找内建的可调用对象
            if len(invoke_chain) == 1: import builtins; invoke_chain.insert(0, "builtins"); head = builtins
            ## 如果首个调用在全局引入范围
            elif invoke_chain[0] in self.context: head = self.context[invoke_chain[0]]
            ## 没有给 globals 或者找不到可调用对象
            else: raise AttributeError(f"找不到可调用对象 {func_sign}" if self.context else f"请确保正确的 context_global")
            for attr in invoke_chain[1:]: head = getattr(head, attr)
            kvargs[name] = head
        return kvargs

    def __exit__(self, *_):
        import inspect as ins
        import argparse
        ## 分析与 enter 时刻 locals 的差异 将 with 内添加的本地变量视作需要解析的参数
        diff_locals = ins.currentframe().f_back.f_locals
        diff_locals = {k: v for k, v in diff_locals.items() if k not in self.record or v != self.record[k]}
        parser:argparse.ArgumentParser = self.parse_args(diff_locals)
        kvargs = dict(parser.parse_args()._get_kwargs())
        kvargs = self.parse_callables(kvargs)
        ins.currentframe().f_back.f_locals.update(kvargs)


if __name__ == "__main__":

    with TimeCost(name="cprint hello", color=True) as tc:
        time.sleep(2)
        cprint("你好", "方式", effect=Effect.italic, sep=" - ", end="\n\n\n")

    cprint(f"A{ctext('H', ForeColor.warn)}Z")

    @deprecated("test deprecated", color=True)
    def test_func(a):
        print(a)

    test_func("test")

