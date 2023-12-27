import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import sys

import os

sys.path.append(os.path.realpath("./src"))
from utils.util import instantiate_dict

# seed_everything()

@hydra.main(version_base=None, config_path="../../config", config_name="config_diffbir")
def main(cfg: DictConfig) -> None:
    config = OmegaConf.load(cfg)
    pl.seed_everything(config.lightning.seed, workers=True)
    
    data_module = instantiate_from_config(config.data)
    model = instantiate_from_config(OmegaConf.load(config.model.config))
    # TODO: resume states saved in checkpoint.
    if config.model.get("resume"):
        load_state_dict(model, torch.load(config.model.resume, map_location="cpu"), strict=True)
    
    callbacks = []
    for callback_config in config.lightning.callbacks:
        callbacks.append(instantiate_from_config(callback_config))
    trainer = pl.Trainer(callbacks=callbacks, **config.lightning.trainer)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()