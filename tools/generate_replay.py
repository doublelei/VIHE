import hydra
from omegaconf import DictConfig, OmegaConf
from VIHE.data.get_dataset import get_dataset

@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    print("Config:\n{}".format(OmegaConf.to_yaml(cfg)))
    device = f"cuda:{cfg.device[0]}"
    tasks = cfg.tasks
    print("Generating Data on {} tasks: {}".format(len(tasks), tasks))
    get_dataset(cfg.dataset, tasks, device)


if __name__ == "__main__":
    main()