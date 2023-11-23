import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys

import os

sys.path.append(os.path.realpath("./src"))
# import core
# from core.utils import seed_everything

# seed_everything()

@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg = OmegaConf.to_object(cfg)
    cfg["module"]["model"] = instantiate(cfg["module"]['model'])
    cfg["module"]["dataset"] = instantiate(cfg["module"]["dataset"])
    cfg["module"] = instantiate(cfg["module"])
    cfg["Trainer"] = instantiate(cfg["Trainer"])

    cfg["Trainer"].predict(cfg["module"], cfg["module"].dataloader)
    
    # cfg["Trainer"].fit(cfg['module']['model'], cfg["module"].dataloader)
    print('It works.')

if __name__ == "__main__":
    main()