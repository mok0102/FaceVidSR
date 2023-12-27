import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys

import os

sys.path.append(os.path.realpath("./src"))
from utils.util import instantiate_dict

# seed_everything()

@hydra.main(version_base=None, config_path="../../config", config_name="config_codeformer")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    # cfg["module"]["model"] = instantiate(cfg["module"]['model'])
    # cfg["module"]["dataset"] = instantiate(cfg["module"]["dataset"])
    # cfg["module"]["optimizer_g"] = instantiate(cfg["module"]["optimizer_g"])
    # cfg["module"]["optimizer_d"] = instantiate(cfg["module"]["optimizer_d"])
    # cfg["module"] = instantiate(cfg["module"])
    # cfg["Trainer"] = instantiate(cfg["Trainer"])
    
    cfg = instantiate_dict(cfg)

    # cfg["Trainer"].predict(cfg["module"], cfg["module"].dataloader)
    
    cfg["Trainer"].fit(cfg['module'])
    print('It works.')

if __name__ == "__main__":
    main()