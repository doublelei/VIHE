# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import shutil
import torch
import clip

from VIHE.data.dataset import create_replay, fill_replay
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset(cfg, tasks, device):
    train_replay_buffer = create_replay(cfg)

    # load pre-trained language model
    clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
    clip_model = clip_model.to(device).eval()

    for task in tasks:  # for each task
        eposides_folder_train = cfg.eposides_folder_train.format(task)
        train_replay_storage_folder = f"{cfg.train_replay_dir}/{task}"

        # if refresh_replay, then remove the existing replay data folder
        if cfg.refresh_replay:
            print("[Info] Remove exisitng replay dataset as requested.", flush=True)
            if os.path.exists(train_replay_storage_folder) and os.path.isdir(
                train_replay_storage_folder
            ):
                shutil.rmtree(train_replay_storage_folder)
                print(f"remove {train_replay_storage_folder}")

        fill_replay(cfg, train_replay_buffer, task, train_replay_storage_folder, eposides_folder_train, clip_model, device)
        
    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode="random",
        num_workers=cfg.num_workers,
        sample_distribution_mode=cfg.sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    
    return train_dataset
