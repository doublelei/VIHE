# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import subprocess

from collections import defaultdict
from contextlib import redirect_stdout

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


import rvt.utils.ddp_utils as ddp_utils
from VIHE.models.utils.mvt import MVT
from VIHE.models.utils.mvtpp import MVTPP
from VIHE.data.get_dataset import get_dataset


from VIHE.logger.tensorboardmanager import TensorboardManager
from VIHE.models.utils.utils import RLBENCH_TASKS, REAL_TASKS, load_agent
from VIHE.models.rvt_agent import RVTAgent, print_loss_log, print_eval_log
from VIHE.models.rvtpp_agent import RVTPPAgent
import hydra
from omegaconf import DictConfig, OmegaConf


def train(agent, dataset, training_iterations, rank=0):
    agent.train()
    log = defaultdict(list)

    data_iter = iter(dataset)
    iter_command = range(training_iterations)

    pbar = tqdm.tqdm(
        iter_command, disable=(rank != 0), position=0, leave=True
    )
    for iteration in pbar:
        raw_batch = next(data_iter)
        batch = {
            k: v.to(agent._device)
            for k, v in raw_batch.items()
            if type(v) == torch.Tensor
        }
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = raw_batch["lang_goal"]
        update_args = {
            "step": iteration,
        }
        if iteration == len(iter_command) - 1:
            eval_log = True
        else:
            eval_log = False

        update_args.update(
            {
                "replay_sample": batch,
                "backprop": True,
                "reset_log": (iteration == 0),
                "eval_log": eval_log,
            }
        )
        return_out = agent.update(**update_args)
        if rank == 0:
            pbar.set_description(
                f' Total loss: {return_out["total_loss"]:.3f}')

    if rank == 0:
        log = print_loss_log(agent)
    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(tasks):
    if tasks == "all":
        tasks = RLBENCH_TASKS
    if tasks == "real_all":
        tasks = REAL_TASKS
    elif tasks == 'split':
        tasks = SPLIT_RLBENCH_TASKS
    return tasks


def get_logdir(cfg):
    if isinstance(cfg.tasks, str):
        tasks = cfg.tasks
    else:
        tasks = ''.join(cfg.tasks)
    log_dir = os.path.join(cfg.log_dir, str(cfg.num_demo),
                           cfg.exp_id, cfg.agent.variant, tasks)
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    log_log_dir = os.path.join(log_dir, "logs")
    os.makedirs(log_log_dir, exist_ok=True)
    return log_dir


def dump_log(cfg):
    with open(f"{cfg.log_dir}/config.yaml", "w") as yaml_file:
        OmegaConf.save(cfg, yaml_file)



def experiment(rank, cfg, devices, port):
    device = devices[rank]
    device = f"cuda:{device}"
    ddp = len(devices) > 1
    if ddp:
        ddp_utils.setup(rank, world_size=len(devices), port=port)
        print(f"Running DDP on rank {rank}.")

    if rank == 0:
        print("Config:\n{}".format(OmegaConf.to_yaml(cfg)))

    cfg.log_dir = get_logdir(cfg)

    tasks = get_tasks(cfg.tasks)
    print("Training on {} tasks: {}".format(len(tasks), tasks))
    print("Log dir: {}".format(cfg.log_dir))

    t_start = time.time()
    train_dataset = get_dataset(cfg.dataset, tasks, device)
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))
    TRAINING_ITERATIONS = int(
        100 // (cfg.dataset.batch_size * len(devices) / 16))
    EPOCHS = cfg.epochs
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    if cfg.agent.name == "rvt":
        
        rvt = MVT(cfg.agent, renderer_device=device).to(device)
        if ddp:
            rvt = DDP(rvt, device_ids=[device], find_unused_parameters=True)

        agent = RVTAgent(
            cfg.agent,
            network=rvt,
            cameras=cfg.dataset.cameras,
            scene_bounds=cfg.dataset.scene_bounds,
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS
        )
        agent.build(training=True, device=device)

    elif cfg.agent.name == "vihe":
        vihe = MVTPP(cfg.agent, renderer_device=device).to(device)
        if ddp:
            vihe = DDP(vihe, device_ids=[device], find_unused_parameters=True)

        agent = RVTPPAgent(
            cfg.agent,
            network=vihe,
            cameras=cfg.dataset.cameras,
            scene_bounds=cfg.dataset.scene_bounds,
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS
        )
        agent.build(training=True, device=device)

    else:
        raise NotImplementedError

    start_epoch = 0
    end_epoch = EPOCHS

    if cfg.agent.resume != "":
        agent_path = cfg.agent.resume
        print(f"Recovering model and checkpoint from {cfg.agent.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1
    if len(devices) > 1:
        dist.barrier()

    if rank == 0:
        # logging unchanged values to reproduce the same setting
        dump_log(cfg)
        tb = TensorboardManager(f"{cfg.log_dir}/logs")

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        if rank == 0 and i % 50 == 0 and i > 0:
            save_agent(agent, f"{cfg.log_dir}/models/model_{i}.pth", i)
            save_agent(agent, f"{cfg.log_dir}/models/model_last.pth", i)

            if cfg.with_eval:
                for eval_task in tasks:
                    command = f"sbatch scripts/eval.sbatch {cfg.num_demo}/{cfg.exp_id} {cfg.agent.variant} {i} {cfg.tasks} {eval_task}"
                    # Run the command
                    process = subprocess.Popen(
                        command.split(), stdout=subprocess.PIPE)

        if i == end_epoch:
            break

        out = train(agent, train_dataset, TRAINING_ITERATIONS, rank)
        if rank == 0:
            tb.update("train", i, out)

    if rank == 0:
        tb.close()
        print("[Finish]")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    port = (random.randint(0, 3000) % 3000) + 27000

    if len(cfg.device) > 1:
        mp.spawn(experiment, args=(cfg, cfg.device, port),
                 nprocs=len(cfg.device), join=True)
    else:
        experiment(0, cfg, cfg.device, port)


if __name__ == "__main__":
    main()
