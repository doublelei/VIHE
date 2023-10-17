# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
RVT PyTorch3D Renderer
"""
from math import ceil, floor
from typing import Optional, Tuple

import torch

from torch import nn
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
)


# source: https://discuss.pytorch.org/t/batched-index-select/9115/6
def batched_index_select(inp, dim, index):
    """
    input: B x * x ... x *
    dim: 0 < scalar
    index: B x M
    """
    views = [inp.shape[0]] + [1 if i != dim else -1 for i in range(1, len(inp.shape))]
    expanse = list(inp.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    return torch.gather(inp, dim, index)


def get_cube_R_T(
    with_scale=False,
    view_dist=1,
):
    """
    Returns camera rotations and translations to render point cloud around a cube
    """
    elev_azim = {
        "top": (0, 0),
        "front": (90, 0),
        "back": (270, 0),
        "left": (0, 90),
        "right": (0, 270),
    }

    elev = torch.tensor([elev for _, (elev, azim) in elev_azim.items()])
    azim = torch.tensor([azim for _, (elev, azim) in elev_azim.items()])

    up = []
    dist = []
    scale = []
    for view in elev_azim:
        if view in ["left", "right"]:
            up.append((0, 0, 1))
        else:
            up.append((0, 1, 0))

        dist.append(view_dist)
        scale.append((1, 1, 1))

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, up=up)
    out = [R, T]
    if with_scale:
        out.append(scale)
    return out


def select_feat_from_hm(
    pt_cam: torch.Tensor, hm: torch.Tensor, pt_cam_wei: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor]:
    """
    :param pt_cam:
        continuous location of point coordinates from where value needs to be
        selected. it is of size [nc, npt, 2], locations in pytorch3d screen
        notations
    :param hm: size [nc, nw, h, w]
    :param pt_cam_wei:
        some predifined weight of size [nc, npt], it is used along with the
        distance weights
    :return:
        tuple with the first element being the wighted average for each point
        according to the hm values. the size is [nc, npt, nw]. the second and
        third elements are intermediate values to be used while chaching
    """
    nc, nw, h, w = hm.shape
    npt = pt_cam.shape[1]
    if pt_cam_wei is None:
        pt_cam_wei = torch.ones([nc, npt]).to(hm.device)

    # giving points outside the image zero weight
    pt_cam_wei[pt_cam[:, :, 0] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 1] < 0] = 0
    pt_cam_wei[pt_cam[:, :, 0] > (w - 1)] = 0
    pt_cam_wei[pt_cam[:, :, 1] > (h - 1)] = 0

    pt_cam = pt_cam.unsqueeze(2).repeat([1, 1, 4, 1])
    # later used for calculating weight
    pt_cam_con = pt_cam.detach().clone()

    # getting discrete grid location of pts in the camera image space
    pt_cam[:, :, 0, 0] = torch.floor(pt_cam[:, :, 0, 0])
    pt_cam[:, :, 0, 1] = torch.floor(pt_cam[:, :, 0, 1])
    pt_cam[:, :, 1, 0] = torch.floor(pt_cam[:, :, 1, 0])
    pt_cam[:, :, 1, 1] = torch.ceil(pt_cam[:, :, 1, 1])
    pt_cam[:, :, 2, 0] = torch.ceil(pt_cam[:, :, 2, 0])
    pt_cam[:, :, 2, 1] = torch.floor(pt_cam[:, :, 2, 1])
    pt_cam[:, :, 3, 0] = torch.ceil(pt_cam[:, :, 3, 0])
    pt_cam[:, :, 3, 1] = torch.ceil(pt_cam[:, :, 3, 1])
    pt_cam = pt_cam.long()  # [nc, npt, 4, 2]
    # since we are taking modulo, points at the edge, i,e at h or w will be
    # mapped to 0. this will make their distance from the continous location
    # large and hence they won't matter. therefore we don't need an explicit
    # step to remove such points
    pt_cam[:, :, :, 0] = torch.fmod(pt_cam[:, :, :, 0], int(w))
    pt_cam[:, :, :, 1] = torch.fmod(pt_cam[:, :, :, 1], int(h))
    pt_cam[pt_cam < 0] = 0

    # getting normalized weight for each discrete location for pt
    # weight based on distance of point from the discrete location
    # [nc, npt, 4]
    pt_cam_dis = 1 / (torch.sqrt(torch.sum((pt_cam_con - pt_cam) ** 2, dim=-1)) + 1e-10)
    pt_cam_wei = pt_cam_wei.unsqueeze(-1) * pt_cam_dis
    _pt_cam_wei = torch.sum(pt_cam_wei, dim=-1, keepdim=True)
    _pt_cam_wei[_pt_cam_wei == 0.0] = 1
    # cached pt_cam_wei in select_feat_from_hm_cache
    pt_cam_wei = pt_cam_wei / _pt_cam_wei  # [nc, npt, 4]

    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    pt_cam = pt_cam.view(nc, 4 * npt, 2)  # [nc, 4 * npt, 2]
    # cached pt_cam in select_feat_from_hm_cache
    pt_cam = (pt_cam[:, :, 1] * w) + pt_cam[:, :, 0]  # [nc, 4 * npt]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, npt, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val, pt_cam, pt_cam_wei


def select_feat_from_hm_cache(
    pt_cam: torch.Tensor,
    hm: torch.Tensor,
    pt_cam_wei: torch.Tensor,
) -> torch.Tensor:
    """
    Cached version of select_feat_from_hm where we feed in directly the
    intermediate value of pt_cam and pt_cam_wei. Look into the original
    function to get the meaning of these values and return type. It could be
    used while inference if the location of the points remain the same.
    """

    nc, nw, h, w = hm.shape
    # transforming indices from 2D to 1D to use pytorch gather
    hm = hm.permute(0, 2, 3, 1).view(nc, h * w, nw)  # [nc, h * w, nw]
    # [nc, 4 * npt, nw]
    pt_cam_val = batched_index_select(hm, dim=1, index=pt_cam)
    # tranforming back each discrete location of point
    pt_cam_val = pt_cam_val.view(nc, -1, 4, nw)
    # summing weighted contribution of each discrete location of a point
    # [nc, npt, nw]
    pt_cam_val = torch.sum(pt_cam_val * pt_cam_wei.unsqueeze(-1), dim=2)
    return pt_cam_val


# unit tests to verify select_feat_from_hm
def test_select_feat_from_hm():
    def get_out(pt_cam, hm):
        nc, nw, d = pt_cam.shape
        nc2, c, h, w = hm.shape
        assert nc == nc2
        assert d == 2
        out = torch.zeros((nc, nw, c))
        for i in range(nc):
            for j in range(nw):
                wx, hx = pt_cam[i, j]
                if (wx < 0) or (hx < 0) or (wx > (w - 1)) or (hx > (h - 1)):
                    out[i, j, :] = 0
                else:
                    coords = (
                        (floor(wx), floor(hx)),
                        (floor(wx), ceil(hx)),
                        (ceil(wx), floor(hx)),
                        (ceil(wx), ceil(hx)),
                    )
                    vals = []
                    total = 0
                    for x, y in coords:
                        val = 1 / (sqrt(((wx - x) ** 2) + ((hx - y) ** 2)) + 1e-10)
                        vals.append(val)
                        total += val

                    vals = [x / total for x in vals]

                    for (x, y), val in zip(coords, vals):
                        out[i, j] += val * hm[i, :, y, x]
        return out

    pt_cam_1 = torch.tensor([[[11.11, 120.1], [37.8, 0.0], [99, 76.5]]])
    hm_1_1 = torch.ones((1, 1, 100, 120))
    hm_1_2 = torch.ones((1, 1, 120, 100))
    out_1 = torch.ones((1, 3, 1))
    out_1[0, 0, 0] = 0

    pt_cam_2 = torch.tensor(
        [
            [[11.11, 12.11], [37.8, 0.0]],
            [[61.00, 12.00], [123.99, 123.0]],
        ]
    )
    hm_2_1 = torch.rand((2, 1, 200, 100))
    hm_2_2 = torch.rand((2, 1, 100, 200))

    test_sets = [
        (pt_cam_1, hm_1_1, out_1),
        (pt_cam_1, hm_1_2, out_1),
        (pt_cam_2, hm_2_1, get_out(pt_cam_2, hm_2_1)),
        (pt_cam_2, hm_2_2, get_out(pt_cam_2, hm_2_2)),
    ]

    for i, test in enumerate(test_sets):
        pt_cam, hm, out = test
        _out, _, _ = select_feat_from_hm(pt_cam, hm)
        out = out.float()
        if torch.all(torch.abs(_out - out) < 1e-5):
            print(f"Passed test {i}, {out}, {_out}")
        else:
            print(f"Failed test {i}, {out}, {_out}")


class PointsRendererWithDepth(nn.Module):
    """
    A class for rendering a batch of points. The class should
    be initialized with a rasterizer and compositor class which each have a forward
    function.
    """

    def __init__(self, rasterizer, compositor, subtract_depth_mean) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor
        self.subtract_depth_mean = subtract_depth_mean

    def to(self, device):
        # Manually move to device rasterizer as the cameras
        # within the class are not of type nn.Module
        self.rasterizer = self.rasterizer.to(device)
        self.compositor = self.compositor.to(device)
        return self

    def forward(self, point_clouds, with_depth=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)
        # meaning of zbuf: https://github.com/facebookresearch/pytorch3d/issues/1147
        if with_depth:
            # (num_img, h, w)
            depth = fragments.zbuf[..., 0]
            _, h, w = depth.shape

            if self.subtract_depth_mean:
                depth_0 = depth == -1
                depth_sum = torch.sum(depth, (1, 2)) + torch.sum(depth_0, (1, 2))
                depth_mean = depth_sum / ((h * w) - torch.sum(depth_0, (1, 2)))
                depth -= depth_mean.unsqueeze(-1).unsqueeze(-1)
                depth[depth_0] = -1

        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        if with_depth:
            images = torch.cat((images, depth.unsqueeze(-1)), dim=-1)

        return images


class BoxRenderer:
    """
    Can be used to render point clouds with fixed cameras and dynamic cameras.
    Code flow: Given the arguments we create a fixed set of cameras around the
    object. We can then render camera images (__call__()), project 3d
    points on the camera images (get_pt_loc_on_img()), project heatmap
    featues onto 3d points (get_feat_frm_hm_cube()) and get 3d point with the
    max heatmap value (get_max_3d_frm_hm_cube()).

    For each of the functions, we can either choose to use the fixed cameras
    which were defined by the argument or create dynamic cameras. The dynamic
    cameras are created by passing the dyn_cam_info argument (explained in
    _get_dyn_cam()).

    For the fixed camera, we optimize the code for projection of heatmap and
    max 3d calculation by caching the values in self._pts, self._fix_pts_cam
    and self._fix_pts_cam_wei.
    """

    def __init__(
        self,
        device,
        img_size,
        radius=0.012,
        points_per_pixel=5,
        compositor="norm",
        with_depth=False,
        real=False,
        dyn_dist=0.001,
        zoom_in=True,
    ):
        """Rendering images form point clouds

        :param device:
        :param img_size:
        :param radius:
        :param points_per_pixel:
        :param compositor:
        """
        self.device = device
        self.img_size = img_size
        self.radius = radius
        self.points_per_pixel = points_per_pixel
        self.compositor = compositor
        self.with_depth = with_depth
        self.real = real
        self.dyn_dist = dyn_dist
        self.zoom_in = zoom_in
        self.init()

    def init(self):
        h, w = self.img_size
        assert h == w

        # used for creating the fixed and dynamic renderers
        self._raster_settings = PointsRasterizationSettings(
            image_size=self.img_size,
            radius=self.radius,
            points_per_pixel=self.points_per_pixel,
            bin_size=0,
        )

        self._local_raster_settings = PointsRasterizationSettings(
            image_size=self.img_size,
            radius=self.radius*2,
            points_per_pixel=self.points_per_pixel*2,
            bin_size=0,
        )

        # self._local2_raster_settings = PointsRasterizationSettings(
        #     image_size=self.img_size,
        #     radius=self.radius*4,
        #     points_per_pixel=self.points_per_pixel*4,
        #     bin_size=0,
        # )

        # used for creating the fixed and dynamic renderers
        if self.compositor == "alpha":
            self._compositor = AlphaCompositor()
        elif self.compositor == "norm":
            self._compositor = NormWeightedCompositor()
        else:
            assert False, f"{self.compositor} is not valid"

        # creates the fixed camera and renderer
        self._fix_cam = None
        self._fix_ren = None

        self._get_fix_cam()
        self._get_fix_ren()

        self.num_img = self.num_fix_cam

        self._pts = None
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None

    def _get_fix_cam(self):
        if self._fix_cam is None:
            R, T, scale = get_cube_R_T(
                with_scale=True,
            )
            assert len(R.shape) == len(T.shape) + 1 == 3
            self._fix_cam = FoVOrthographicCameras(
                device=self.device,
                R=R,
                T=T,
                znear=0.01,
                scale_xyz=scale,
            )
            self.num_fix_cam = len(R)

        return self._fix_cam

    def _get_fix_ren(self):
        if self._fix_ren is None:
            rasterizer = PointsRasterizer(
                cameras=self._get_fix_cam(), raster_settings=self._raster_settings
            )
            self._fix_ren = PointsRendererWithDepth(
                rasterizer=rasterizer, compositor=self._compositor, subtract_depth_mean=True
            )

        return self._fix_ren

    def _get_dyn_cam(self, stage):
        """
        :param dyn_cam_info: tuple of (R, T, scale, K) where R is array of shape
            (num_dyn_cam, 3, 3), T (num_dyn_cam, 3), scale (num_dyn_cam) and K
            (num-dyn_cam, 4, 4)
        """

        if self.zoom_in==1:
            if stage==0: max_range = 1
            elif stage==1: max_range = 0.5
            elif stage==2: max_range = 0.25
            else:
                assert False, f"Invalid stage {stage}"
        elif self.zoom_in==2:
            if stage==0: max_range = 1
            elif stage==1: max_range = 0.5
            elif stage==2: max_range = 0.125
            else:
                assert False, f"Invalid stage {stage}"
        else:
            max_range = 1


        R, T, scale = get_cube_R_T(
            with_scale=True,
            view_dist=self.dyn_dist,
        )

        dyn_cam = FoVOrthographicCameras(
            device=self.device, R=R, T=T, 
            znear=0.001, zfar=max_range,
            min_x=-max_range, max_x=max_range,
            min_y=-max_range, max_y=max_range,
            scale_xyz=scale
        )
        return dyn_cam

    def _get_dyn_ren(self, stage):
        """
        :param dyn_cam_info: tuple of (R, T, scale, K) where R is array of shape
            (num_dyn_cam, 3, 3), T (num_dyn_cam, 3), scale (num_dyn_cam) and K
            (num-dyn_cam, 4, 4)
        """
        rasterizer = PointsRasterizer(
            cameras=self._get_dyn_cam(stage),
            raster_settings=self._local_raster_settings,
        )
        if self.dyn_dist == 1.:
            subtract_depth_mean = True
        else:
            subtract_depth_mean = False
        dyn_ren = PointsRendererWithDepth(
            rasterizer=rasterizer, compositor=self._compositor, subtract_depth_mean=subtract_depth_mean
        )
        return dyn_ren

    def img_norm(self, img):
        """
        some post processing of the images
        """
        if self.compositor == "norm":
            if self.with_depth:
                if img[..., :-1].max() > 1:
                    assert img[..., :-1].max() < 1.001, img[..., :-1].max()
                    img[..., :-1] /= img[..., :-1].max()
            else:
                if img.max() > 1:
                    assert img.max() < 1.001
                    img /= img.max()
        return img

    @torch.no_grad()
    def __call__(self, pc, feat, fix_cam=True, stage1_W_E_H_4x4=None, stage2_W_E_H_4x4=None):
        """
        :param pc: torch.Tensor  (num_point, 3)
        :param feat: torch.Tensor (num_point, num_feat)
        :param fix_cam: whether to render fixed cameras of not
        :param dyn_cam_info:
            Either:
                - None: dynamic cameras are not rendered
                - dyn_cam_info: a single element tuple of tuple of elements
                    described in  _get_dyn_cam()
        :return: (num_img,  h, w, num_feat)
        """
        assert pc.shape[-1] == 3
        assert len(pc.shape) == 2
        assert len(feat.shape) == 2
        assert isinstance(pc, torch.Tensor)
        # assert (stage1_dyn_cam_info is None) or (
        #     isinstance(stage1_dyn_cam_info, (list, tuple))
        #     and isinstance(stage1_dyn_cam_info[0], tuple)
        # ), stage1_dyn_cam_info
        pc = [pc]
        feat = [feat]
        img = []

        if fix_cam:
            p3d_pc = Pointclouds(points=pc, features=feat)
            p3d_pc = p3d_pc.extend(self.num_fix_cam)
            renderer = self._get_fix_ren()
            fix_img = renderer(p3d_pc, with_depth=self.with_depth)
            img = self.img_norm(fix_img)

        else:
            img = None

        if not stage1_W_E_H_4x4 is None:
            # invert the transformation
            H_E_W_4x4 = torch.inverse(stage1_W_E_H_4x4)
            # pc_stage_1 = torch.matmul(H_E_W_4x4[None, :3, :3], pc[0].unsqueeze(-1)).squeeze(-1) + H_E_W_4x4[:3, 3]

            pc_stage_1 = torch.matmul(H_E_W_4x4[:3, :3], pc[0].T).T + H_E_W_4x4[:3, 3]
            pc_stage_1 = [pc_stage_1]
            p3d_pc = Pointclouds(points=pc_stage_1, features=feat)
            p3d_pc = p3d_pc.extend(self.num_fix_cam)
            renderer = self._get_dyn_ren(stage=1)
            dyn_img = renderer(p3d_pc, with_depth=self.with_depth)
            stage1_local_img = self.img_norm(dyn_img)

        else:
            stage1_local_img = None

        if not stage2_W_E_H_4x4 is None:
            # invert the transformation
            H_E_W_4x4 = torch.inverse(stage2_W_E_H_4x4)
            pc_stage_2 = torch.matmul(H_E_W_4x4[None, :3, :3], pc[0].unsqueeze(-1)).squeeze(-1) + H_E_W_4x4[:3, 3]
            pc_stage_2 = [pc_stage_2]
            p3d_pc = Pointclouds(points=pc_stage_2, features=feat)
            p3d_pc = p3d_pc.extend(self.num_fix_cam)
            renderer = self._get_dyn_ren(stage=2)
            dyn_img = renderer(p3d_pc, with_depth=self.with_depth)
            stage2_local_img = self.img_norm(dyn_img)

        else:
            stage2_local_img = None

        return img, stage1_local_img, stage2_local_img

    @torch.no_grad()
    def get_pt_loc_on_img(self, pt, W_E_H_4x4, stage):
        """
        returns the location of a point on the image of the cameras
        :param pt: torch.Tensor of shape (bs, np, 3)
        :param fix_cam: same as __call__
        :param dyn_cam_info: same as __call__
        :returns: the location of the pt on the image. this is different from the
            camera screen coordinate system in pytorch3d. the difference is that
            pytorch3d camera screen projects the point to [0, 0] to [H, W]; while the
            index on the img is from [0, 0] to [H-1, W-1]. We verified that
            the to transform from pytorch3d camera screen point to img we have to
            subtract (1/H, 1/W) from the pytorch3d camera screen coordinate.
        :return type: torch.Tensor of shape (bs, np, self.num_img, 2)
        """
        assert len(pt.shape) == 3
        assert pt.shape[-1] == 3
        bs, np = pt.shape[0:2]

        pt_img = []
        if stage==0:
            fix_cam = self._get_fix_cam()
            # (num_cam, bs * np, 2)
            pt_scr = fix_cam.transform_points_screen(
                pt.view(-1, 3), image_size=self.img_size
            )[..., 0:2]

            if len(fix_cam) == 1:
                pt_scr = pt_scr.unsqueeze(0)

            pt_scr = torch.transpose(pt_scr, 0, 1)
            # transform from camera screen to image index
            h, w = self.img_size
            # (bs * np, num_img, 2)
            fix_pt_img = pt_scr - torch.tensor((1 / w, 1 / h)).to(pt_scr.device)
            fix_pt_img = fix_pt_img.view(bs, np, len(fix_cam), 2)
            pt_img.append(fix_pt_img)
        else:
            dyn_cam = self._get_dyn_cam(stage=stage)

            H_E_W_4x4 = torch.inverse(W_E_H_4x4)
            H_pt = torch.matmul(H_E_W_4x4[:, :3, :3], pt.squeeze(1).unsqueeze(-1)).squeeze(-1) + H_E_W_4x4[:,:3, 3]
            
            # import pdb; pdb.set_trace()
            # pc_stage_1 = torch.matmul(H_E_W_4x4[:3, :3], pt.T).T + H_E_W_4x4[:3, 3]

            # R_3x3 = W_E_H_4x4[:, :3, :3].transpose(1,2)
            # t = W_E_H_4x4[:, :3, 3:4]
            # t_ = pt.squeeze(1).unsqueeze(-1) - t
            # p_ = torch.matmul(R_3x3, t_).squeeze(-1)
            # import pdb; pdb.set_trace()

            # H_pt = torch.matmul(H_E_W_4x4[:, :3, :3], pt.squeeze(1).unsqueeze(-1)).squeeze(-1) + H_E_W_4x4[:,:3, 3]

            pt_scr = dyn_cam.transform_points_screen(
                H_pt.view(-1, 3), image_size=self.img_size
            )[..., 0:2]

            # pt_scr = dyn_cam.transform_points_screen(
            #     H_pt.view(-1, 3), image_size=self.img_size
            # )
            # print(pt_scr[...,2])
            # pt_scr = pt_scr[..., 0:2]

            if self.num_fix_cam == 1:
                pt_scr = pt_scr.unsqueeze(0)

            pt_scr = torch.transpose(pt_scr, 0, 1)
            # transform from camera screen to image index
            h, w = self.img_size
            # (bs * np, num_img, 2)
            dyn_pt_img = pt_scr - torch.tensor((1 / w, 1 / h)).to(pt_scr.device)
            dyn_pt_img = dyn_pt_img.view(bs, np, self.num_fix_cam, 2)
            pt_img.append(dyn_pt_img)

        pt_img = torch.cat(pt_img, 2)

        return pt_img

    @torch.no_grad()
    def get_feat_frm_hm_cube(self, hm, stage, fix_cam=True, W_E_H_4x4=None):
        """
        :param hm: torch.Tensor of (1, num_img, h, w)
        :return: tupe of ((num_img, h^3, 1), (h^3, 3))
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == self.num_img
        assert self.img_size == (h, w)

        res = self.img_size[0]
        pts = torch.linspace(-1 + (1 / res), 1 - (1 / res), res).to(hm.device)
        pts = torch.cartesian_prod(pts, pts, pts)

        if self.zoom_in==1:
            if stage == 0: pass
            elif stage == 1: pts = pts * 0.5
            elif stage == 2: pts = pts * 0.25
            else: assert False, f"Invalid stage {stage}"
        elif self.zoom_in==2:
            if stage == 0: pass
            elif stage == 1: pts = pts * 0.5
            elif stage == 2: pts = pts * 0.125
        else:
            pass

        if stage == 0:
            pass
        else:
            pts = torch.matmul(W_E_H_4x4[:, :3, :3], pts.unsqueeze(-1)).squeeze(-1) + W_E_H_4x4[0, :3, 3]

        pts_hm = []
        if fix_cam:
            if self._fix_pts_cam is None:
                # (np, nc, 2)
                pts_img = self.get_pt_loc_on_img(pts.unsqueeze(0), W_E_H_4x4, 0).squeeze(0)
                # (nc, np, bs)
                fix_pts_hm, pts_cam, pts_cam_wei = select_feat_from_hm(
                    pts_img.transpose(0, 1), hm.transpose(0, 1)[0 : self.num_fix_cam]
                )
                self._fix_pts_cam = pts_cam
                self._fix_pts_cam_wei = pts_cam_wei
            else:
                pts_cam = self._fix_pts_cam
                pts_cam_wei = self._fix_pts_cam_wei
                fix_pts_hm = select_feat_from_hm_cache(
                    pts_cam, hm.transpose(0, 1)[0 : self.num_fix_cam], pts_cam_wei
                )
            pts_hm.append(fix_pts_hm)

        else:
            assert stage > 0
            pts_img = self.get_pt_loc_on_img(
                pts.unsqueeze(0),
                W_E_H_4x4,
                stage,
            ).squeeze(0)
            dyn_pts_hm, _, _ = select_feat_from_hm(
                pts_img.transpose(0, 1), hm.transpose(0, 1) #[self.num_fix_cam :]
            )
            pts_hm.append(dyn_pts_hm)

        pts_hm = torch.cat(pts_hm, 0)
        
        return pts_hm, pts

    @torch.no_grad()
    def get_max_3d_frm_hm_cube(self, hm, stage, fix_cam=True, W_E_H_4x4=None):
        """
        given set of heat maps, return the 3d location of the point with the
            largest score, assumes the points are in a cube [-1, 1]. This function
            should be used  along with the render. For standalone version look for
            the other function with same name in the file.
        :param hm: (1, nc, h, w)
        :param fix_cam:
        :param dyn_cam_info:
        :return: (1, 3)
        """
        x, nc, h, w = hm.shape
        assert x == 1
        assert nc == self.num_img
        assert self.img_size == (h, w)

        pts_hm, pts = self.get_feat_frm_hm_cube(hm, stage, fix_cam, W_E_H_4x4)
        # (bs, np, nc)
        pts_hm = pts_hm.permute(2, 1, 0)
        # (bs, np)
        pts_hm = torch.mean(pts_hm, -1)
        # (bs)
        ind_max_pts = torch.argmax(pts_hm, -1)
        return pts[ind_max_pts]

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        del self._pts
        del self._fix_pts_cam
        del self._fix_pts_cam_wei
        self._pts = None
        self._fix_pts_cam = None
        self._fix_pts_cam_wei = None
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()