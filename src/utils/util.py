import os
import random
import torch
import numpy as np
from natsort import natsorted
from glob import glob
from hydra.utils import instantiate

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def make_dir(save_path:str, name: str) -> None:
    logdirname = natsorted([dirname for dirname in glob("./lightning_logs/version_*")])[-1] #to find the latest version of lightninglog
    folderpath = os.path.join(save_path, logdirname, name)
    os.makedirs(folderpath, exist_ok=True)
    return folderpath

def instantiate_dict(cfg)->dict:
    #print(cfg)
    for key, value in cfg.items():
        if isinstance(value, dict) and "_target_" in value:
            cfg[key] = instantiate_dict(value)
            if key == "optimizer" or key=="optimizer_g" or key=="optimizer_d":
                if isinstance(cfg["model"], dict):
                    cfg["model"] = instantiate(instantiate_dict(cfg["model"]))
                cfg[key]["params"] = cfg["model"].parameters()
            if key == "scheduler":
                cfg["scheduler"]["optimizer"] = cfg["optimizer"]
            cfg[key] = instantiate(cfg[key])
        elif isinstance(value, str):
            if value == "True" or value == "False":
                cfg[key] = bool(value if value=="True" else "")
        
    return cfg

        


