# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Utility function for MVT
"""
import pdb
import sys

import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP


RLBENCH_TASKS = [
    "put_item_in_drawer",
    "reach_and_drag",
    "turn_tap",
    "slide_block_to_color_target",
    "open_drawer",
    "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter",
    "put_money_in_safe",
    "push_buttons",
    "close_jar",
    "stack_blocks",
    "place_cups",
    "place_wine_at_rack_location",
    "light_bulb_in",
    "sweep_to_dustpan_of_size",
    "insert_onto_square_peg",
    "meat_off_grill",
    "stack_cups",
]

REAL_TASKS = ["put_item_in_shelf",
              "stack_cube",
              "place_marker",
              "press_sanitizer",
              "put_cup_in_hanger",
              "put_item_in_drawer"
              ]


def place_pc_in_cube(
    pc, app_pc=None, with_mean_or_bounds=True, scene_bounds=None, no_op=False
):
    """
    calculate the transformation that would place the point cloud (pc) inside a
        cube of size (2, 2, 2). The pc is centered at mean if with_mean_or_bounds
        is True. If with_mean_or_bounds is False, pc is centered around the mid
        point of the bounds. The transformation is applied to point cloud app_pc if
        it is not None. If app_pc is None, the transformation is applied on pc.
    :param pc: pc of shape (num_points_1, 3)
    :param app_pc:
        Either
        - pc of shape (num_points_2, 3)
        - None
    :param with_mean_or_bounds:
        Either:
            True: pc is centered around its mean
            False: pc is centered around the center of the scene bounds
    :param scene_bounds: [x_min, y_min, z_min, x_max, y_max, z_max]
    :param no_op: if no_op, then this function does not do any operation
    """
    if no_op:
        if app_pc is None:
            app_pc = torch.clone(pc)

        return app_pc, lambda x: x

    if with_mean_or_bounds:
        assert scene_bounds is None
    else:
        assert not (scene_bounds is None)
    if with_mean_or_bounds:
        pc_mid = (torch.max(pc, 0)[0] + torch.min(pc, 0)[0]) / 2
        x_len, y_len, z_len = torch.max(pc, 0)[0] - torch.min(pc, 0)[0]
    else:
        x_min, y_min, z_min, x_max, y_max, z_max = scene_bounds
        pc_mid = torch.tensor(
            [
                (x_min + x_max) / 2,
                (y_min + y_max) / 2,
                (z_min + z_max) / 2,
            ]
        ).to(pc.device)
        x_len, y_len, z_len = x_max - x_min, y_max - y_min, z_max - z_min

    scale = 2 / max(x_len, y_len, z_len)
    if app_pc is None:
        app_pc = torch.clone(pc)
    app_pc = (app_pc - pc_mid) * scale

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        return (x / scale) + pc_mid

    return app_pc, rev_trans


def trans_pc(pc, loc, sca):
    """
    change location of the center of the pc and scale it
    :param pc:
        either:
        - tensor of shape(b, num_points, 3)
        - tensor of shape(b, 3)
        - list of pc each with size (num_points, 3)
    :param loc: (b, 3 )
    :param sca: 1 or (3)
    """
    assert len(loc.shape) == 2
    assert loc.shape[-1] == 3
    if isinstance(pc, list):
        assert all([(len(x.shape) == 2) and (x.shape[1] == 3) for x in pc])
        pc = [sca * (x - y) for x, y in zip(pc, loc)]
    elif isinstance(pc, torch.Tensor):
        assert len(pc.shape) in [2, 3]
        assert pc.shape[-1] == 3
        pc = sca * (pc - loc)
    else:
        assert False

    # reverse transformation to obtain app_pc in original frame
    def rev_trans(x):
        assert isinstance(x, torch.Tensor)
        return (x / sca) + loc

    return pc, rev_trans


def add_uni_noi(x, u):
    """
    adds uniform noise to a tensor x. output is tensor where each element is
    in [x-u, x+u]
    :param x: tensor
    :param u: float
    """
    assert isinstance(u, float)
    # move noise in -1 to 1
    noise = (2 * torch.rand(*x.shape, device=x.device)) - 1
    x = x + (u * noise)
    return x


def generate_hm_from_pt(pt, res, sigma, thres_sigma_times=3):
    """
    Pytorch code to generate heatmaps from point. Points with values less than
    thres are made 0
    :type pt: torch.FloatTensor of size (num_pt, 2)
    :type res: int or (int, int)
    :param sigma: the std of the gaussian distribition. if it is -1, we
        generate a hm with one hot vector
    :type sigma: float
    :type thres: float
    """
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    _hmx = torch.arange(0, resy).to(pt.device)
    _hmx = _hmx.view([1, resy]).repeat(resx, 1).view([resx, resy, 1])
    _hmy = torch.arange(0, resx).to(pt.device)
    _hmy = _hmy.view([resx, 1]).repeat(1, resy).view([resx, resy, 1])
    hm = torch.cat([_hmx, _hmy], dim=-1)
    hm = hm.view([1, resx, resy, 2]).repeat(num_pt, 1, 1, 1)
    pt = pt.view([num_pt, 1, 1, 2])
    hm = torch.exp(-1 * torch.sum((hm - pt) ** 2, -1) / (2 * (sigma**2)))
    thres = np.exp(-1 * (thres_sigma_times**2) / 2)
    hm[hm < thres] = 0.0

    hm /= torch.sum(hm, (1, 2), keepdim=True) + 1e-6
    # TODO: make a more efficient version
    if sigma == -1:
        _hm = hm.view(num_pt, resx * resy)
        hm = torch.zeros((num_pt, resx * resy), device=hm.device)
        temp = torch.arange(num_pt).to(hm.device)
        hm[temp, _hm.argmax(-1)] = 1

    return hm


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h)
    grid = torch.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = torch.cat([torch.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = torch.cat([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = torch.einsum('m,d->md', pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], axis=1)
    return emb


def grid_coordinate_pixel_coordinate(grid_coordinate, grid_size=20):
    """
    real_coordinate: Tensor of shape (BS, 2) representing real coordinates
    grid_size: Size of the original grid (default 20)
    cell_size: Size of each cell in the grid in pixels (default 11)
    returns: Tensor of shape (BS, 2) representing pixel coordinates
    # """
    # transform to pixeled grid
    pixel_coords = grid_coordinate * \
        (grid_size + 1) / grid_size - 0.5  # Scale to grid_size
    return pixel_coords


def get_2d_pos_embedding_for_pixels(pixels, embed_dim, grid_size=20, cell_size=11):
    """
    pixels: Tensor of shape (BS, 2) representing pixel coordinates
    embed_dim: Dimension of the position embedding
    grid_size: Size of the original grid (default 20)
    cell_size: Size of each cell in the grid in pixels (default 11)
    returns: Tensor of shape (BS, embed_dim) representing position embeddings
    """
    # Normalize the pixel coordinates to the grid
    grid_coords = pixels.float() / cell_size

    # transform to pixeled grid
    pixel_coords = grid_coordinate_pixel_coordinate(
        grid_coords, grid_size=grid_size)
    # grid_coords = grid_coords * (grid_size + 1) / grid_size - 0.5  # Scale to grid_size
#     grid_coords = grid_coords * (grid_size - 1) / grid_size  # Scale to grid_size

    # Split coordinates into separate dimensions
    pos_h, pos_w = pixel_coords.split(1, dim=-1)

    # Obtain the 1D position embeddings for both dimensions
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, pos_h.squeeze())
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, pos_w.squeeze())

    # Concatenate the embeddings for both dimensions
    emb = torch.cat([emb_h, emb_w], dim=-1)

    return emb


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a tensor of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float).to(pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.flatten()  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(replay_sample, cameras):
    obs, pcds = [], []
    for n in cameras:
        rgb = stack_on_channel(replay_sample["%s_rgb" % n])
        pcd = stack_on_channel(replay_sample["%s_point_cloud" % n])

        rgb = _norm_rgb(rgb)

        obs.append(
            [rgb, pcd]
        )  # obs contains both rgb and pointcloud (used in ARM for other baselines)
        pcds.append(pcd)  # only pointcloud
    return obs, pcds


def get_pc_img_feat(obs, pcd, bounds=None):
    """
    preprocess the data in the peract to our framework
    """
    # obs, pcd = peract_utils._preprocess_inputs(batch)
    bs = obs[0][0].shape[0]
    # concatenating the points from all the cameras
    # (bs, num_points, 3)
    pc = torch.cat([p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd], 1)
    # pc = torch.cat([p for p in pcd], 1)
    _img_feat = [o[0] for o in obs]
    img_dim = _img_feat[0].shape[1]
    # (bs, num_points, 3)
    img_feat = torch.cat(
        [p.permute(0, 2, 3, 1).reshape(bs, -1, img_dim) for p in _img_feat], 1
    )
    # img_feat = torch.cat([p for p in _img_feat], 1)
    img_feat = (img_feat + 1) / 2

    # x_min, y_min, z_min, x_max, y_max, z_max = bounds
    # inv_pnt = (
    #     (pc[:, :, 0] < x_min)
    #     | (pc[:, :, 0] > x_max)
    #     | (pc[:, :, 1] < y_min)
    #     | (pc[:, :, 1] > y_max)
    #     | (pc[:, :, 2] < z_min)
    #     | (pc[:, :, 2] > z_max)
    # )

    # # TODO: move from a list to a better batched version
    # pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    # img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]

    return pc, img_feat


def move_pc_in_bound(pc, img_feat, bounds, no_op=False):
    """
    :param no_op: no operation
    """
    if no_op:
        return pc, img_feat

    x_min, y_min, z_min, x_max, y_max, z_max = bounds
    inv_pnt = (
        (pc[:, :, 0] < x_min)
        | (pc[:, :, 0] > x_max)
        | (pc[:, :, 1] < y_min)
        | (pc[:, :, 1] > y_max)
        | (pc[:, :, 2] < z_min)
        | (pc[:, :, 2] > z_max)
        | torch.isnan(pc[:, :, 0])
        | torch.isnan(pc[:, :, 1])
        | torch.isnan(pc[:, :, 2])
    )

    # TODO: move from a list to a better batched version
    pc = [pc[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    img_feat = [img_feat[i, ~_inv_pnt] for i, _inv_pnt in enumerate(inv_pnt)]
    return pc, img_feat


def get_num_feat(cfg):
    num_feat = cfg.num_rotation_classes * 3
    num_feat += 4
    return num_feat


def load_agent(agent_path, agent=None, only_epoch=False):
    checkpoint = torch.load(agent_path, map_location="cpu")
    epoch = checkpoint["epoch"]

    if not only_epoch:
        if hasattr(agent, "_q"):
            model = agent._q
        elif hasattr(agent, "_network"):
            model = agent._network
        optimizer = agent._optimizer
        lr_sched = agent._lr_sched

        if isinstance(model, DDP):
            model = model.module

        try:
            model.load_state_dict(checkpoint["model_state"])
        except RuntimeError:
            try:
                print(
                    "WARNING: loading states in mvt1. "
                    "Be cautious if you are using a two stage network."
                )
                if hasattr(model, "svl1"):
                    single_model = model.svl1
                else:
                    single_model = model.mvt1
                single_model.load_state_dict(checkpoint["model_state"])
            except RuntimeError:
                print(
                    "WARNING: loading states with strick=False! "
                    "KNOW WHAT YOU ARE DOING!!"
                )
                model.load_state_dict(checkpoint["model_state"], strict=False)
        if "optimizer_state" in checkpoint:
            loaded_state_dict = checkpoint["optimizer_state"]
            optimizer_state_dict = optimizer.state_dict()
            # compare_state_dicts(loaded_state_dict, optimizer_state_dict)

            print(len(loaded_state_dict['param_groups'][0]['params']))
            print(len(optimizer_state_dict['param_groups'][0]['params']))

            compare_param_groups(checkpoint["optimizer_state"], optimizer)
            compare_optimizer_state_dicts(
                checkpoint["optimizer_state"], optimizer)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(
                "WARNING: No optimizer_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

        if "lr_sched_state" in checkpoint:
            lr_sched.load_state_dict(checkpoint["lr_sched_state"])
        else:
            print(
                "WARNING: No lr_sched_state in checkpoint" "KNOW WHAT YOU ARE DOING!!"
            )

    return epoch


def compare_state_dicts(loaded_state_dict, model_state_dict):
    loaded_keys = set(loaded_state_dict.keys())
    model_keys = set(model_state_dict.keys())

    common_keys = model_keys.intersection(loaded_keys)
    if len(common_keys) == len(model_keys):
        print("Keys are matching perfectly!")
    else:
        print(f"Common keys ({len(common_keys)}):")
        for key in common_keys:
            print(key)

        only_in_model = model_keys - loaded_keys
        if only_in_model:
            print(f"\nKeys only in current model ({len(only_in_model)}):")
            for key in only_in_model:
                print(key)

        only_in_loaded = loaded_keys - model_keys
        if only_in_loaded:
            print(f"\nKeys only in loaded state_dict ({len(only_in_loaded)}):")
            for key in only_in_loaded:
                print(key)

    # Shape comparison for common keys
    for key in common_keys:
        if loaded_state_dict[key].shape != model_state_dict[key].shape:
            print(f"Mismatch found for key {key}: loaded shape is {loaded_state_dict[key].shape}, "
                  f"model shape is {model_state_dict[key].shape}")
        else:
            print(f"Key {key}: Shapes match perfectly!")


def compare_optimizer_state_dicts(loaded_state_dict, optimizer):
    optimizer_state_dict = optimizer.state_dict()

    loaded_keys = set(loaded_state_dict.keys())
    optimizer_keys = set(optimizer_state_dict.keys())

    common_keys = optimizer_keys.intersection(loaded_keys)
    if len(common_keys) == len(optimizer_keys):
        print("Keys are matching perfectly!")
    else:
        print(f"Common keys ({len(common_keys)}):")
        for key in common_keys:
            print(key)

        only_in_optimizer = optimizer_keys - loaded_keys
        if only_in_optimizer:
            print(
                f"\nKeys only in current optimizer ({len(only_in_optimizer)}):")
            for key in only_in_optimizer:
                print(key)

        only_in_loaded = loaded_keys - optimizer_keys
        if only_in_loaded:
            print(f"\nKeys only in loaded state_dict ({len(only_in_loaded)}):")
            for key in only_in_loaded:
                print(key)


def compare_param_groups(loaded_state_dict, optimizer):
    loaded_param_groups = loaded_state_dict['param_groups']
    optimizer_param_groups = optimizer.state_dict()['param_groups']

    print(
        f"Number of parameter groups in loaded state dict: {len(loaded_param_groups)}")
    print(
        f"Number of parameter groups in optimizer: {len(optimizer_param_groups)}")

    if len(loaded_param_groups) != len(optimizer_param_groups):
        print("The number of parameter groups does not match!")
    else:
        print("The number of parameter groups matches perfectly.")



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