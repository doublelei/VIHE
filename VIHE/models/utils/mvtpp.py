import torch

from torch import nn
from VIHE.models.utils.renderer_pp import BoxRenderer
from VIHE.models.utils.mvtpp_single import MVT as MVTSingle
from VIHE.models.utils.renderer import get_cube_R_T


class MVTPP(nn.Module):
    def __init__(
        self,
        cfg,
        renderer_device="cuda:0",
    ):
        """MultiView Transfomer"""
        super().__init__()
        self.cfg = cfg
        self.renderer = BoxRenderer(
            device=renderer_device,
            img_size=(cfg.img_size, cfg.img_size),
            with_depth=cfg.add_depth,
        )
        self.num_img = self.renderer.num_img
        self.mvt1 = MVTSingle(cfg, renderer=self.renderer)

    def get_pt_loc_on_img(self, pt, W_E_H_4x4, stage):
        """
        :param pt: point for which location on image is to be found. the point
            shoud be in the same reference frame as wpt_local (see forward()),
            even for mvt2
        :param out: output from mvt, when using mvt2, we also need to provide the
            origin location where where the point cloud needs to be shifted
            before estimating the location in the image
        """
        assert len(pt.shape) == 3
        bs, np, x = pt.shape
        assert x == 3

        out = self.mvt1.get_pt_loc_on_img(pt, W_E_H_4x4=W_E_H_4x4, stage=stage)

        return out

    def get_wpt(self, out, W_E_H_4x4, stage):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        :param y_q: refer to the definition in mvt_single.get_wpt
        """

        wpt = self.mvt1.get_wpt(out, W_E_H_4x4, stage)
        return wpt

    def render(self, pc, img_feat, img_aug, stage1_W_E_H_4x4=None, stage2_W_E_H_4x4=None):
        mvt = self.mvt1

        with torch.no_grad():
            img, stage1_local_img, stage2_local_img = [], [], []

            for (_pc, _img_feat, _stage1_W_E_H_4x4, _stage2_W_E_H_4x4) in zip(
                pc, img_feat, stage1_W_E_H_4x4, stage2_W_E_H_4x4
            ):
                _img, _stage1_local_img, _stage2_local_img = self.renderer(
                    _pc,
                    torch.cat((_pc, _img_feat), dim=-1),
                    fix_cam=True,
                    stage1_W_E_H_4x4=_stage1_W_E_H_4x4,
                    stage2_W_E_H_4x4=_stage2_W_E_H_4x4,
                )
                img.append(_img)
                stage1_local_img.append(_stage1_local_img)
                stage2_local_img.append(_stage2_local_img)

            def post_process_img(img):
                img = torch.stack(img, 0)
                img = img.permute(0, 1, 4, 2, 3)

                # for visualization purposes
                if mvt.cfg.add_corr:
                    mvt.img = img[:, :, 3:].clone().detach()
                else:
                    mvt.img = img.clone().detach()

                # image augmentation
                if img_aug != 0:
                    stdv = img_aug * torch.rand(1, device=img.device)
                    # values in [-stdv, stdv]
                    noise = stdv * \
                        ((2 * torch.rand(*img.shape, device=img.device)) - 1)
                    img = torch.clamp(img + noise, -1, 1)

                if mvt.cfg.add_pixel_loc:
                    bs, _, _, h, w = img.shape

                    # pixel_loc = mvt.pixel_loc.to(img.device)
                    pixel_loc = mvt.pixel_loc
                    pixel_loc = pixel_loc[:, :, :h, :w]
                    img = torch.cat(
                        (img, pixel_loc.unsqueeze(0).repeat(bs, 1, 1, 1, 1)), dim=2
                    )
                return img

            img = post_process_img(img)
            if stage1_local_img[0] is not None:
                stage1_local_img = post_process_img(stage1_local_img)
            if stage2_local_img[0] is not None:
                stage2_local_img = post_process_img(stage2_local_img)

        return img, stage1_local_img, stage2_local_img

    def verify_inp(
        self,
        pc,
        img_feat,
        proprio,
        lang_emb,
        img_aug,
    ):
        if not self.training:
            # no img_aug when not training
            assert img_aug == 0

        bs = len(pc)
        assert bs == len(img_feat)

        for _pc, _img_feat in zip(pc, img_feat):
            np, x1 = _pc.shape
            np2, x2 = _img_feat.shape

            assert np == np2
            assert x1 == 3
            assert x2 == self.cfg.img_feat_dim

        if self.cfg.add_proprio:
            bs3, x3 = proprio.shape
            assert bs == bs3
            assert (
                x3 == self.cfg.proprio_dim
            ), "Does not support proprio of shape {proprio.shape}"
        else:
            assert proprio is None, "Invalid input for proprio={proprio}"

    def get_dyn_cam_info(self, local_ctr):
        """
        :param local_ctr: tensor of shape (bs, 3)
        """

        R, T, scale = get_cube_R_T(True)
        R = R.to(self.renderer_device)
        T = T.to(self.renderer_device)
        dyn_cam_info = []

        for i in range(len(local_ctr)):
            rot = []
            trans = []
            r = R  # [i]
            t = -local_ctr[i] @ R

            rot.append(r)
            trans.append(t)

            r = torch.stack(rot).flatten(0, 1)
            t = torch.stack(trans).flatten(0, 1)

            s = torch.ones(t.shape[0], 3).to(self.renderer_device)
            k = None
            dyn_cam_info.append((r, t, s, k))
        return dyn_cam_info

    def forward(
        self,
        pc,
        img_feat,
        proprio=None,
        lang_emb=None,
        img_aug=0,
        stage1_W_E_H_4x4=None,
        stage2_W_E_H_4x4=None,
        **kwargs,
    ):
        """
        :param pc: list of tensors, each tensor of shape (num_points, 3)
        :param img_feat: list tensors, each tensor of shape
            (bs, num_points, img_feat_dim)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        self.verify_inp(pc, img_feat, proprio, lang_emb, img_aug)

        img, stage1_local_img, stage2_local_img = self.render(
            pc,
            img_feat,
            img_aug,
            stage1_W_E_H_4x4=stage1_W_E_H_4x4,
            stage2_W_E_H_4x4=stage2_W_E_H_4x4,
        )

        out = self.mvt1(img=img, stage1_local_img=stage1_local_img, stage2_local_img=stage2_local_img,
                        stage1_W_E_H_4x4=stage1_W_E_H_4x4, stage2_W_E_H_4x4=stage2_W_E_H_4x4,
                        proprio=proprio, lang_emb=lang_emb, **kwargs)

        return out, img, stage1_local_img, stage2_local_img

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()


