from VIHE.data.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n{}".format(OmegaConf.to_yaml(cfg)))
    device = f"cuda:{cfg.device}"
    tasks = cfg.dataset.tasks
    
    print("Generating Data on {} tasks: {}".format(len(tasks), tasks))
    
    get_dataset(cfg.dataset, device)


if __name__ == "__main__":
    main()