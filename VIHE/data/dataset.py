import os
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from VIHE.data.utils import get_stored_demo, keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs
EPISODE_FOLDER = "episode%d"
# the pkl file that contains language goals for each demonstration
VARIATION_DESCRIPTIONS_PKL = "variation_descriptions.pkl"


def create_replay(cfg):
    image_width, image_height = cfg.image_size

    # low_dim_state
    observation_elements = []
    observation_elements.append(ObservationElement(
        "low_dim_state", (cfg.low_dim_state, ), np.float32))

    # rgb, depth, point cloud, intrinsics, extrinsics
    element_shapes_types = [
        ("_rgb", (3, image_width, image_height), np.float32),
        ("_depth", (1, image_width, image_height), np.float32),
        ("_point_cloud", (3, image_width, image_height), np.float32),
        ("_camera_extrinsics", (4, 4), np.float32),
        ("_camera_intrinsics", (3, 3), np.float32),
    ]

    # Add observation elements for each camera
    for cname in cfg.cameras:
        observation_elements.extend(
            ObservationElement(f"{cname}{suffix}", shape, dtype)
            for suffix, shape, dtype in element_shapes_types
        )

    # discretized translation, discretized rotation, discrete ignore collision, 6-DoF gripper pose, and pre-trained language embeddings

    observation_elements.extend([
        ReplayElement("trans_action_indicies",
                      (cfg.trans_indicies_size,), np.int32),
        ReplayElement("rot_grip_action_indicies",
                      (cfg.rot_and_grip_indicies_size,), np.int32),
        ReplayElement("ignore_collisions",
                      (cfg.ignore_collisions_size,), np.int32),
        ReplayElement("gripper_pose", (cfg.gripper_pose_size,), np.float32),
        ReplayElement("lang_goal_embs", (cfg.max_token_seq_len,
                      cfg.lang_emb_dim), np.float32),
        ReplayElement("lang_goal", (1,), object),
    ])

    extra_replay_elements = [
        ReplayElement("demo", (), bool),
        ReplayElement("keypoint_idx", (), int),
        ReplayElement("episode_idx", (), int),
        ReplayElement("keypoint_frame", (), int),
        ReplayElement("next_keypoint_frame", (), int),
        ReplayElement("sample_frame", (), int),
    ]

    replay_buffer = (
        UniformReplayBuffer(  # all tuples in the buffer have equal sample weighting
            disk_saving=cfg.disk_saving,
            batch_size=cfg.batch_size,
            timesteps=cfg.timesteps,
            replay_capacity=cfg.replay_capacity,
            action_shape=(8,),
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
        )
    )
    return replay_buffer


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1: Observation,
    obs_tm1: Observation,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(
            obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )


# extract CLIP language features for goal string
def _clip_encode_text(clip_model, text):
    x = clip_model.token_embedding(text).type(
        clip_model.dtype
    )  # [batch_size, n_ctx, d_model]

    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = clip_model.ln_final(x).type(clip_model.dtype)

    emb = x.clone()
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)
          ] @ clip_model.text_projection

    return x, emb


# add individual data points to a replay
def _add_keypoints_to_replay(
    cfg,
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs: Observation,
    demo: Demo,
    episode_keypoints: List[int],
    next_keypoint_idx: int,
    description: str = "",
    lang_embs=None,
):
    prev_action = None
    obs = inital_obs
    for k in range(next_keypoint_idx, len(episode_keypoints)):
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            cfg.scene_bounds,
            cfg.voxel_size,
            cfg.rotation_resolution,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0
        obs_dict = extract_obs(
            obs, cfg.cameras, t=k - next_keypoint_idx, episode_length=25)

        obs_dict["lang_goal_embs"] = lang_embs
        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    obs_dict_tp1 = extract_obs(obs_tp1, cfg.cameras, t=k + 1 -
                               next_keypoint_idx, prev_action=prev_action, episode_length=25)
    obs_dict_tp1["lang_goal_embs"] = lang_embs
    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)


def fill_replay(cfg, replay, task, task_replay_storage_folder, data_path, clip_model, device):
    disk_exist = False
    if replay._disk_saving:
        print(task_replay_storage_folder)
        if os.path.exists(task_replay_storage_folder):
            logging.info(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                )
            )
            disk_exist = True
        else:
            logging.info("\t saving to disk: %s", task_replay_storage_folder)
            os.makedirs(task_replay_storage_folder, exist_ok=True)
    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        for d_idx in range(cfg.num_demos):
            logging.info("Filling %s demo %d" % (task, d_idx))
            
            demo = get_stored_demo(data_path=data_path, index=d_idx)

            # get language goal from disk
            varation_descs_pkl_file = os.path.join(
                data_path, EPISODE_FOLDER % d_idx, VARIATION_DESCRIPTIONS_PKL
            )
            with open(varation_descs_pkl_file, "rb") as f:
                descs = pickle.load(f)
            desc = descs[0]
            tokens = clip.tokenize([desc]).numpy()
            token_tensor = torch.from_numpy(tokens).to(device)
            with torch.no_grad():
                _, lang_embs = _clip_encode_text(
                    clip_model, token_tensor)
                lang_embs = lang_embs.squeeze(0).cpu().numpy()
            episode_keypoints = keypoint_discovery(demo, method=cfg.keypoint_method)
            next_keypoint_idx = 0

            if cfg.replay_buffer_sample_policy == 'uniform':
                sample_frames = range(
                    0, len(demo), cfg.replay_buffer_sample_freq)
            else:
                raise ValueError('Unknown replay_buffer_sample: %s' %
                                 cfg.replay_buffer_sample_policy)

            for i in sample_frames:
                if not cfg.demo_augmentation and i > 0:
                    break
                obs = demo[i]

                while (
                    next_keypoint_idx < len(episode_keypoints)
                    and i >= episode_keypoints[next_keypoint_idx]
                ):
                    next_keypoint_idx += 1
                if next_keypoint_idx == len(episode_keypoints):
                    break
                _add_keypoints_to_replay(
                    cfg,
                    replay,
                    task,
                    task_replay_storage_folder,
                    d_idx,
                    i,
                    obs,
                    demo,
                    episode_keypoints,
                    next_keypoint_idx,
                    desc,
                    lang_embs,
                )

        # save TERMINAL info in replay_info.npy
        task_idx = replay._task_index[task]
        with open(
            os.path.join(task_replay_storage_folder, "replay_info.npy"), "wb"
        ) as fp:
            np.save(
                fp,
                replay._store["terminal"][
                    replay._task_replay_start_index[
                        task_idx
                    ]: replay._task_replay_start_index[task_idx]
                    + replay._task_add_count[task_idx].value
                ],
            )

        print("Replay filled with demos.")
