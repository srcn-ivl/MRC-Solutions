from typing import List, Callable, Union

import shutil as sh
import sys
import os


FILT_IMG_DAT = lambda f: f.lower().endswith((".jpg", ".png"))
FILT_SUF = lambda fmts: lambda f: f.lower().endswith(fmts) 


def sorted_ord_file_list(
        path:str, 
        fill:int=16
    ) -> List[str]:

    """
    获取序号文件目录的序号排序绝对路径列表

    ["1.x", "10.x", ... , "99.x"] --> ["{pwd}/1.x", "{pwd}/2.x", ... , "{pwd}/99.x"]
    """

    return [os.path.join(path, f) for f in sorted(os.listdir(path), key=lambda n: n.split(".")[0].zfill(fill))]


def op_dir(
        path:str, 
        rec:int=1
    ) -> str:

    """for _ in range(rec): path = os.path.dirname(path)"""

    for _ in range(rec):
        path = os.path.dirname(path)
    return path


def op_base(
        path:str, 
        rec:int=1
    ) -> str:

    """os.path.basename(path)"""
    
    if rec == 1: return os.path.basename(path)
    else: return path.replace(op_dir(path, rec), "")[1:]


def op_pbase(
        path:str, 
        rec:int=1
    ) -> str:

    """os.path.basename(for _ in range(rec): path = os.path.dirname(path))"""

    for _ in range(rec):
        path = os.path.dirname(path)
    return os.path.basename(path)


def op_tbase(
        path:str, 
        rec:int=1
    ) -> str:
    """path{op_dir(path, rec) -> Nan}"""
    return path.replace(op_dir(path, rec), "")


def op_real(path:str) -> str:
    """os.path.realpath(path)"""
    return os.path.realpath(path)


def op_abs(path:str) -> str:
    """os.path.abspath(path)"""
    return os.path.abspath(path)


def cwd() -> str:
    """os.path.realpath(os.getcwd())"""
    return op_real(os.getcwd())


def j(
        base:str, 
        *p:str
    ) -> str:
    """os.path.join(base, *p)"""
    return os.path.join(base, *p)


def relf(
        _file_:str,
        *p:str
    ) -> str:
    """os.path.join(os.path.dirname(file), *p)"""
    return os.path.join(os.path.dirname(_file_), *p)


def irelf(*p:str) -> str:
    """inspect invoker path, and relf(..)"""
    import inspect as ins
    _file_ = ins.stack()[1].filename
    return relf(_file_, *p)


def isfile(path:str) -> bool:
    """os.path.isfile(path)"""
    return os.path.isfile(path)


def isdir(path:str) -> bool:
    """os.path.isdir(path)"""
    return os.path.isdir(path)


def islink(path:str) -> bool:
    """os.path.islink(path)"""
    return os.path.islink(path)


def exists(path:str) -> bool:
    """os.path.exists(path)"""
    return os.path.exists(path)


def cmd(command:str) -> int:
    """os.system(command)"""
    return os.system(command)


def ls(
        path:str, 
        filter:Callable[[str],bool]=None
    ) -> List[str]:

    """
        LiSt path  

        os.listdir(path) (/filter{p})
    """

    lst = os.listdir(path)
    if filter is None: return lst
    else: return [f for f in lst if filter(f)]


def als(
        path:str, 
        filter:Callable[[str],bool]=None
    ) -> List[str]:

    """
        Absolutely LiSt path  

        [os.path.join(path, p) for p in os.listdir(path)] (/filter{os.path.join(path, p)})
    """

    abs_ls = [j(path, p) for p in ls(path)]
    if filter is None: return abs_ls
    return [ap for ap in abs_ls if filter(ap)]


def lf(
        *patterns:str,
        recursive=False
) -> List[str]:
    """glob.glob(*patterns, recursive=recursive)"""
    import glob
    return glob.glob(*patterns, recursive=recursive)


def cp(
        src:str,
        dst:str
    ) -> None:
    """shutil.copy(src, dst)"""
    sh.copy(src, dst)


def mds(path:str) -> None:
    """os.makedirs(path, exist_ok=True)"""
    return os.makedirs(path, exist_ok=True)


def nm(
        path:str, 
        full:bool=True
    ) -> str:
    """os.path.splitext(path)[0]"""
    if not full: path = op_base(path)
    return os.path.splitext(path)[0]


def renm(
        src:str, 
        dst:str
    ) -> None:
    """os.renames(src, dst)"""
    os.renames(src, dst)


def rm(path:str) -> None:
    """! remove path"""
    if isfile(path): os.remove(path)
    else: sh.rmtree(path)


def fmt(path:str) -> str:
    """os.path.splitext(path)[-1]"""
    return os.path.splitext(path)[-1].lower()


def refmt(
        path:str, 
        fmt:str
    ) -> str:
    """os.path.splitext(path)[0] + fmt"""
    return os.path.splitext(path)[0] + fmt


def check_same_workdir(
        sfile1:str, 
        sfile2:str=cwd()
    ) -> bool:

    """Flag of files have same work directory"""
    
    sfile1 = op_dir(op_real(sfile1))
    return sfile1 == sfile2


def ensure_dirs(
        path:str, 
        is_file:bool=False, 
        empty:bool=False
    ) -> str:

    """Make sure (an {empty}) directories exist"""

    if is_file is None: is_file = "." in op_base(path)
    dir_path = os.path.dirname(path) if is_file else path
    if empty and exists(path): sh.rmtree(path) 
    os.makedirs(dir_path, exist_ok=not empty)
    return path


def read_json(json_path:str) -> Union[list, dict]:
    import json
    with open(json_path, "r", encoding="utf8") as reader:
        data = json.load(reader)
    return data


def dump_json(
        dump_contents:Union[list, dict], 
        dump_path:str
    ) -> None:
    import json
    with open(dump_path, "w", encoding="utf8") as writer:
        json.dump(dump_contents, writer, indent=4, ensure_ascii=False)


class WorkDir():

    """临时切换工作目录上下文"""

    def __init__(
            self, 
            work_dir:str=os.getcwd()
        ) -> None:
        self.ori_wd = os.getcwd()
        self.temp_wd = work_dir
        if self.temp_wd: sys.path.insert(0, work_dir)

    def __enter__(self):
        if not self.temp_wd: return
        os.chdir(self.temp_wd)
        return self
    
    def __exit__(self, *_):
        if not self.temp_wd: return
        os.chdir(self.ori_wd)
        if self.temp_wd in sys.path:
            sys.path.remove(self.temp_wd)


if __name__ == "__main__":

    # print(sorted_ord_file_list(""))
    # print(check_same_workdir())
    print(fmt("/home/pack.JPG"))

