# Adapted from https://github.com/stepjam/RLBench/blob/master/rlbench/utils.py


import logging
import os
import pickle
import numpy as np

from PIL import Image
from typing import List

from rlbench.backend.observation import Observation
from rlbench.backend.utils import image_to_float_array
from rlbench.demo import Demo

from pyrep.objects import VisionSensor


# constants
EPISODE_FOLDER = 'episode%d'

CAMERA_FRONT = 'front'
CAMERA_LS = 'left_shoulder'
CAMERA_RS = 'right_shoulder'
CAMERA_WRIST = 'wrist'
CAMERAS = [CAMERA_FRONT, CAMERA_LS, CAMERA_RS, CAMERA_WRIST]


IMAGE_RGB = 'rgb'
IMAGE_DEPTH = 'depth'
IMAGE_TYPES = [IMAGE_RGB, IMAGE_DEPTH]
IMAGE_FORMAT = '%d.png'
LOW_DIM_PICKLE = 'low_dim_obs.pkl'
VARIATION_NUMBER_PICKLE = 'variation_number.pkl'

DEPTH_SCALE = 2**24 - 1
# functions

REMOVE_KEYS = ['joint_velocities', 'joint_positions', 'joint_forces',
               'gripper_open', 'gripper_pose',
               'gripper_joint_positions', 'gripper_touch_forces',
               'task_low_dim_state', 'misc']


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
        i < (len(demo) - 2) and
        (obs.gripper_open == demo[i + 1].gripper_open and
         obs.gripper_open == demo[i - 1].gripper_open and
         demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo,
                       stopping_delta=0.1,
                       method='heuristic') -> List[int]:
    episode_keypoints = []
    if method == 'heuristic':
        prev_gripper_open = demo[0].gripper_open
        stopped_buffer = 0
        for i, obs in enumerate(demo):
            stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            # If change in gripper, or end of episode.
            last = i == (len(demo) - 1)
            if i != 0 and (obs.gripper_open != prev_gripper_open or
                           last or stopped):
                episode_keypoints.append(i)
            prev_gripper_open = obs.gripper_open
        if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
                episode_keypoints[-2]:
            episode_keypoints.pop(-2)
        logging.debug('Found %d keypoints.' % len(episode_keypoints),
                      episode_keypoints)
        return episode_keypoints

    elif method == 'random':
        # Randomly select keypoints.
        episode_keypoints = np.random.choice(
            range(len(demo)),
            size=20,
            replace=False)
        episode_keypoints.sort()
        return episode_keypoints

    elif method == 'fixed_interval':
        # Fixed interval.
        episode_keypoints = []
        segment_length = len(demo) // 20
        for i in range(0, len(demo), segment_length):
            episode_keypoints.append(i)
        return episode_keypoints
    elif method == 'fixed_real':
        for i, obs in enumerate(demo):
            if obs.is_keypoint:
                print(i, obs.is_keypoint)
                episode_keypoints.append(i)
        return episode_keypoints
    else:
        raise NotImplementedError


def get_stored_demo(data_path, index):
    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)

    # low dim pickle file
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        # print(f, episode_path, LOW_DIM_PICKLE, os.path.join(episode_path, LOW_DIM_PICKLE))
        # import pdb; pdb.set_trace()
        obs = pickle.load(f)

    # variation number
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)

    num_steps = len(obs)
    for i in range(num_steps):
        obs[i].front_rgb = np.array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].left_shoulder_rgb = np.array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_LS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].right_shoulder_rgb = np.array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_RS, IMAGE_RGB), IMAGE_FORMAT % i)))
        obs[i].wrist_rgb = np.array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_RGB), IMAGE_FORMAT % i)))

        obs[i].front_depth = image_to_float_array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_FRONT)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_FRONT)]
        obs[i].front_depth = near + obs[i].front_depth * (far - near)

        obs[i].left_shoulder_depth = image_to_float_array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_LS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_LS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_LS)]
        obs[i].left_shoulder_depth = near + \
            obs[i].left_shoulder_depth * (far - near)

        obs[i].right_shoulder_depth = image_to_float_array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_RS, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_RS)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_RS)]
        obs[i].right_shoulder_depth = near + \
            obs[i].right_shoulder_depth * (far - near)

        obs[i].wrist_depth = image_to_float_array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_WRIST, IMAGE_DEPTH), IMAGE_FORMAT % i)), DEPTH_SCALE)
        near = obs[i].misc['%s_camera_near' % (CAMERA_WRIST)]
        far = obs[i].misc['%s_camera_far' % (CAMERA_WRIST)]
        obs[i].wrist_depth = near + obs[i].wrist_depth * (far - near)

        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth,
                                                                                        obs[i].misc['front_camera_extrinsics'],
                                                                                        obs[i].misc['front_camera_intrinsics'])
        obs[i].left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].left_shoulder_depth,
                                                                                                obs[i].misc['left_shoulder_camera_extrinsics'],
                                                                                                obs[i].misc['left_shoulder_camera_intrinsics'])
        obs[i].right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].right_shoulder_depth,
                                                                                                 obs[i].misc['right_shoulder_camera_extrinsics'],
                                                                                                 obs[i].misc['right_shoulder_camera_intrinsics'])
        obs[i].wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].wrist_depth,
                                                                                        obs[i].misc['wrist_camera_extrinsics'],
                                                                                        obs[i].misc['wrist_camera_intrinsics'])

    return obs


# functions
def get_stored_demo_real(data_path, index, size=(400, 300)):
    # print(123, size)
    episode_path = os.path.join(data_path, EPISODE_FOLDER % index)

    # low dim pickle file
    with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
        print(f, episode_path, LOW_DIM_PICKLE,
              os.path.join(episode_path, LOW_DIM_PICKLE))
        # import pdb; pdb.set_trace()
        obs = pickle.load(f)

    # variation number
    with open(os.path.join(episode_path, VARIATION_NUMBER_PICKLE), 'rb') as f:
        obs.variation_number = pickle.load(f)

    num_steps = len(obs)
    for i in range(num_steps):
        # print("obs_indx", i)
        obs[i].front_rgb = np.array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_RGB), IMAGE_FORMAT % i)))
        ori_size = obs[i].front_rgb.shape[:2]
        # obs[i].front_rgb = cv2.resize(obs[i].front_rgb, size)

        obs[i].front_depth = image_to_float_array(Image.open(os.path.join(
            episode_path, '%s_%s' % (CAMERA_FRONT, IMAGE_DEPTH), IMAGE_FORMAT % i)))

        # obs[i].front_rgb = cv2.resize(obs[i].front_rgb, size)
        # obs[i].front_depth = cv2.resize(obs[i].front_depth, size, interpolation = cv2.INTER_NEAREST)
        # intrinsic = modify_intrinsic_matrix(obs[i].misc['front_camera_intrinsics'], ori_size, (size[1], size[0]))
        obs[i].front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(obs[i].front_depth,
                                                                                        obs[i].misc['front_camera_extrinsics'],
                                                                                        obs[i].misc['front_camera_intrinsics'])
        pc = obs[i].front_point_cloud
        rgb = obs[i].front_rgb
        depth = obs[i].front_depth

        # import pdb; pdb.set_trace()

        inv_pnt = (
            (pc[:, :, 0] < x_min)
            | (pc[:, :, 0] > x_max)
            | (pc[:, :, 1] < y_min)
            | (pc[:, :, 1] > y_max)
            | (pc[:, :, 2] < z_min)
            | (pc[:, :, 2] > z_max)
            | np.isnan(pc[:, :, 0])
            | np.isnan(pc[:, :, 1])
            | np.isnan(pc[:, :, 2])
        )
        # import pdb; pdb.set_trace()
        # TODO: move from a list to a better batched version
        pc = [pc[j, ~_inv_pnt] for j, _inv_pnt in enumerate(inv_pnt)]
        pc = np.concatenate(pc, axis=0)
        smaple_inds = np.random.choice(pc.shape[0], 160000, replace=True)
        pc = pc[smaple_inds]
        pc = pc.reshape(400, 400, 3)

        rgb = [rgb[j, ~_inv_pnt] for j, _inv_pnt in enumerate(inv_pnt)]
        rgb = np.concatenate(rgb, axis=0)
        rgb = rgb[smaple_inds]
        rgb = rgb.reshape(400, 400, 3)

        depth = [depth[j, ~_inv_pnt] for j, _inv_pnt in enumerate(inv_pnt)]
        depth = np.concatenate(depth, axis=0)
        depth = depth[smaple_inds]
        depth = depth.reshape(400, 400)

        obs[i].front_point_cloud = pc
        obs[i].front_rgb = rgb
        obs[i].front_depth = depth
        # import pdb; pdb.set_trace()
    return obs


def modify_intrinsic_matrix(K, original_shape, new_shape):
    """
    Modify the intrinsic matrix for an image resize.

    Args:
    - K (np.array): Original 3x3 intrinsic matrix.
    - original_shape (tuple): Original image shape as (height, width).
    - new_shape (tuple): New image shape as (height, width).

    Returns:
    - np.array: Modified 3x3 intrinsic matrix.
    """
    # Calculate the scaling factors
    y_scale = new_shape[0] / original_shape[0]
    x_scale = new_shape[1] / original_shape[1]

    # Create a scaling matrix
    scale_matrix = np.array([
        [x_scale, 0, 0],
        [0, y_scale, 0],
        [0, 0, 1]
    ])

    # Multiply the original intrinsic matrix by the scaling matrix
    K_new = np.dot(scale_matrix, K)

    return K_new


def extract_obs(obs: Observation,
                cameras,
                t: int = 0,
                channels_last: bool = False,
                episode_length: int = 10,
                relative=False):
    obs.joint_velocities = None
    grip_mat = obs.gripper_matrix
    grip_pose = obs.gripper_pose
    joint_pos = obs.joint_positions
    obs.gripper_pose = None
    obs.gripper_matrix = None
    obs.wrist_camera_matrix = None
    obs.joint_positions = None
    if obs.gripper_joint_positions is not None:
        obs.gripper_joint_positions = np.clip(
            obs.gripper_joint_positions, 0., 0.04)

    obs_dict = vars(obs)
    obs_dict = {k: v for k, v in obs_dict.items() if v is not None}

    # print(obs.gripper_open, obs.gripper_joint_positions)
    robot_state = np.array([
        obs.gripper_open,
        obs.gripper_joint_positions[0],
        obs.gripper_joint_positions[1]])
    # remove low-level proprioception variables that are not needed
    obs_dict = {k: v for k, v in obs_dict.items()
                if k not in REMOVE_KEYS}

    if not channels_last:
        # swap channels from last dim to 1st dim
        obs_dict = {k: np.transpose(
            v, [2, 0, 1]) if v.ndim == 3 else np.expand_dims(v, 0)
            for k, v in obs_dict.items() if type(v) == np.ndarray or type(v) == list}
    else:
        # add extra dim to depth data
        obs_dict = {k: v if v.ndim == 3 else np.expand_dims(v, -1)
                    for k, v in obs_dict.items()}
    # print(robot_state)
    obs_dict['low_dim_state'] = np.array(robot_state, dtype=np.float32)

    # binary variable indicating if collisions are allowed or not while planning paths to reach poses
    obs_dict['ignore_collisions'] = np.array(
        [obs.ignore_collisions], dtype=np.float32)
    for (k, v) in [(k, v) for k, v in obs_dict.items() if 'point_cloud' in k]:
        obs_dict[k] = v.astype(np.float32)

    for camera_name in cameras:
        obs_dict['%s_camera_extrinsics' %
                 camera_name] = obs.misc['%s_camera_extrinsics' % camera_name]
        obs_dict['%s_camera_intrinsics' %
                 camera_name] = obs.misc['%s_camera_intrinsics' % camera_name]

    # add timestep to low_dim_state
    if relative:
        time = t / float(episode_length)
    else:
        time = (1. - (t / float(episode_length - 1))) * 2. - 1.
    obs_dict['low_dim_state'] = np.concatenate(
        [obs_dict['low_dim_state'], [time]]).astype(np.float32)

    obs.gripper_matrix = grip_mat
    obs.joint_positions = joint_pos
    obs.gripper_pose = grip_pose

    return obs_dict
