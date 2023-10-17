# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange

from VIHE.models.utils.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    cache_fn,
    DenseBlock,
    FeedForward,
)


class MVT(nn.Module):
    def __init__(
        self,
        cfg,
        renderer=None,
    ):
        
        super().__init__()
        self.cfg = cfg

        assert not renderer is None
        self.renderer = renderer
        self.num_img = self.renderer.num_img

        # patchified input dimensions
        spatial_size = cfg.img_size // cfg.img_patch_size  # 128 / 8 = 16

        if self.cfg.add_proprio:
            # 64 img features + 64 proprio features
            self.input_dim_before_seq = self.cfg.im_channels * 2
        else:
            self.input_dim_before_seq = self.cfg.im_channels

        # learnable positional encoding
        if cfg.add_lang:
            self.lang_emb_dim, self.lang_max_seq_len = cfg.lang_dim, cfg.lang_len
        else:
            self.lang_emb_dim, self.lang_max_seq_len = 0, 0

        if cfg.pe_fix:
            num_pe_token = spatial_size**2 * self.num_img
        else:
            num_pe_token = self.lang_max_seq_len + (spatial_size**2 * self.num_img)
        self.pos_encoding = nn.Parameter(
            torch.randn(
                1,
                num_pe_token,
                self.input_dim_before_seq,
            )
        )

        inp_img_feat_dim = cfg.img_feat_dim
        if cfg.add_corr:
            inp_img_feat_dim += 3
        if cfg.add_pixel_loc:
            inp_img_feat_dim += 3
            self.pixel_loc = torch.zeros(
                (self.num_img, 3, self.cfg.img_size, self.cfg.img_size)
            )
            self.pixel_loc[:, 0, :, :] = (
                torch.linspace(-1, 1, self.num_img).unsqueeze(-1).unsqueeze(-1)
            )
            self.pixel_loc[:, 1, :, :] = (
                torch.linspace(-1, 1, cfg.img_size).unsqueeze(0).unsqueeze(-1)
            )
            self.pixel_loc[:, 2, :, :] = (
                torch.linspace(-1, 1, cfg.img_size).unsqueeze(0).unsqueeze(0)
            )
        if cfg.add_depth:
            inp_img_feat_dim += 1

        # img input preprocessing encoder
        self.input_preprocess = Conv2DBlock(
            inp_img_feat_dim,
            cfg.im_channels,
            kernel_sizes=1,
            strides=1,
            norm=None,
            activation=cfg.activation,
        )
        inp_pre_out_dim = self.cfg.im_channels

        if self.cfg.add_proprio:
            # proprio preprocessing encoder
            self.proprio_preprocess = DenseBlock(
                self.cfg.proprio_dim,
                self.cfg.im_channels,
                norm="group",
                activation=cfg.activation,
            )

        self.patchify = Conv2DBlock(
            inp_pre_out_dim,
            cfg.im_channels,
            kernel_sizes=cfg.img_patch_size,
            strides=cfg.img_patch_size,
            norm="group",
            activation=cfg.activation,
            padding=0,
        )

        # lang preprocess
        if cfg.add_lang:
            self.lang_preprocess = DenseBlock(
                self.lang_emb_dim,
                cfg.im_channels * 2,
                norm="group",
                activation=cfg.activation,
            )

        self.fc_bef_attn = DenseBlock(
            self.input_dim_before_seq,
            cfg.attn_dim,
            norm=None,
            activation=None,
        )
        self.fc_aft_attn = DenseBlock(
            cfg.attn_dim,
            self.input_dim_before_seq,
            norm=None,
            activation=None,
        )

        get_attn_attn = lambda: PreNorm(
            cfg.attn_dim,
            Attention(
                cfg.attn_dim,
                heads=cfg.attn_heads,
                dim_head=cfg.attn_dim_head,
                dropout=cfg.attn_dropout,
            ),
        )
        get_attn_ff = lambda: PreNorm(cfg.attn_dim, FeedForward(cfg.attn_dim))
        get_attn_attn, get_attn_ff = map(cache_fn, (get_attn_attn, get_attn_ff))
        # self-attention layers
        self.layers = nn.ModuleList([])
        cache_args = {"_cache": cfg.weight_tie_layers}

        for _ in range(cfg.depth):
            self.layers.append(
                nn.ModuleList([get_attn_attn(**cache_args), get_attn_ff(**cache_args)])
            )

        self.up0 = Conv2DUpsampleBlock(
            self.input_dim_before_seq,
            self.cfg.im_channels,
            kernel_sizes=cfg.img_patch_size,
            strides=cfg.img_patch_size,
            norm=None,
            activation=cfg.activation,
        )

        final_inp_dim = self.cfg.im_channels + inp_pre_out_dim

        # final layers
        self.final = Conv2DBlock(
            final_inp_dim,
            self.cfg.im_channels,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=cfg.activation,
        )

        self.trans_decoder = Conv2DBlock(
            self.cfg.final_dim,
            1,
            kernel_sizes=3,
            strides=1,
            norm=None,
            activation=None,
        )

        feat_out_size = cfg.feat_dim
        feat_fc_dim = 0
        feat_fc_dim += self.input_dim_before_seq
        feat_fc_dim += self.cfg.final_dim

        self.feat_fc = nn.Sequential(
            nn.Linear(self.num_img * feat_fc_dim, feat_fc_dim),
            nn.ReLU(),
            nn.Linear(feat_fc_dim, feat_fc_dim // 2),
            nn.ReLU(),
            nn.Linear(feat_fc_dim // 2, feat_out_size),
        )

    def get_pt_loc_on_img(self, pt, dyn_cam_info):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """
        pt_img = self.renderer.get_pt_loc_on_img(
            pt, fix_cam=True, dyn_cam_info=dyn_cam_info
        )
        return pt_img

    def forward(
        self,
        img,
        proprio=None,
        lang_emb=None,
        **kwargs,
    ):
        """
        :param img: tensor of shape (bs, num_img, img_feat_dim, h, w)
        :param proprio: tensor of shape (bs, priprio_dim)
        :param lang_emb: tensor of shape (bs, lang_len, lang_dim)
        :param img_aug: (float) magnitude of augmentation in rgb image
        """

        bs, num_img, img_feat_dim, h, w = img.shape
        num_pat_img = h // self.cfg.img_patch_size
        assert num_img == self.num_img
        # assert img_feat_dim == self.img_feat_dim
        assert h == w == self.cfg.img_size

        img = img.view(bs * num_img, img_feat_dim, h, w)
        # preprocess
        # (bs * num_img, im_channels, h, w)
        d0 = self.input_preprocess(img)

        # (bs * num_img, im_channels, h, w) ->
        # (bs * num_img, im_channels, h / img_patch_strid, w / img_patch_strid) patches
        ins = self.patchify(d0)
        # (bs, im_channels, num_img, h / img_patch_strid, w / img_patch_strid) patches
        ins = (
            ins.view(
                bs,
                num_img,
                self.cfg.im_channels,
                num_pat_img,
                num_pat_img,
            )
            .transpose(1, 2)
            .clone()
        )

        # concat proprio
        _, _, _d, _h, _w = ins.shape
        if self.cfg.add_proprio:
            p = self.proprio_preprocess(proprio)  # [B,4] -> [B,64]
            p = p.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, _d, _h, _w)
            ins = torch.cat([ins, p], dim=1)  # [B, 128, num_img, np, np]

        # channel last
        ins = rearrange(ins, "b d ... -> b ... d")  # [B, num_img, np, np, 128]

        # save original shape of input for layer
        ins_orig_shape = ins.shape

        # flatten patches into sequence
        ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        # add learable pos encoding
        # only added to image tokens
        if self.cfg.pe_fix:
            ins += self.pos_encoding

        # append language features as sequence
        num_lang_tok = 0
        if self.cfg.add_lang:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128]

        # add learable pos encoding
        if not self.cfg.pe_fix:
            ins = ins + self.pos_encoding

        x = self.fc_bef_attn(ins)
        if self.cfg.self_cross_ver == 0:
            # self-attention layers
            for self_attn, self_ff in self.layers:
                x = self_attn(x) + x
                x = self_ff(x) + x

        elif self.cfg.self_cross_ver == 1:
            lx, imgx = x[:, :num_lang_tok], x[:, num_lang_tok:]

            # within image self attention
            imgx = imgx.reshape(bs * num_img, num_pat_img * num_pat_img, -1)
            for self_attn, self_ff in self.layers[: len(self.layers) // 2]:
                imgx = self_attn(imgx) + imgx
                imgx = self_ff(imgx) + imgx

            imgx = imgx.view(bs, num_img * num_pat_img * num_pat_img, -1)
            x = torch.cat((lx, imgx), dim=1)
            # cross attention
            for self_attn, self_ff in self.layers[len(self.layers) // 2 :]:
                x = self_attn(x) + x
                x = self_ff(x) + x

        else:
            assert False

        # append language features as sequence
        if self.cfg.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]
        x = self.fc_aft_attn(x)

        # reshape back to orginal size
        x = x.view(bs, *ins_orig_shape[1:-1], x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, num_pat_img, num_pat_img
            )
        )

        u0 = self.up0(x)
        u0 = torch.cat([u0, d0], dim=1)
        u = self.final(u0)

        # translation decoder
        trans = self.trans_decoder(u).view(bs, self.num_img, h, w)

        hm = F.softmax(trans.detach().view(bs, self.num_img, h * w), 2).view(
            bs * self.num_img, 1, h, w
        )

        _feat = torch.sum(hm * u, dim=[2, 3])
        _feat = _feat.view(bs, -1)
        feat.append(_feat)
        feat = torch.cat(feat, dim=-1)
        feat = self.feat_fc(feat)

        out = {"trans": trans, "feat": feat}
        return out

    def get_wpt(self, out, dyn_cam_info, y_q=None):
        """
        Estimate the q-values given output from mvt
        :param out: output from mvt
        """
        nc = self.num_img
        h = w = self.cfg.img_size
        bs = out["trans"].shape[0]

        q_trans = out["trans"].view(bs, nc, h * w)
        hm = torch.nn.functional.softmax(q_trans, 2)
        hm = hm.view(bs, nc, h, w)

        if dyn_cam_info is None:
            dyn_cam_info_itr = (None,) * bs
        else:
            dyn_cam_info_itr = dyn_cam_info

        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                fix_cam=True,
                dyn_cam_info=dyn_cam_info_itr[i : i + 1]
                if not (dyn_cam_info_itr[i] is None)
                else None,
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)

        assert y_q is None
        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
