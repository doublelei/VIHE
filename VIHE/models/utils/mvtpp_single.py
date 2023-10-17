# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

from math import ceil

import torch
import torch.nn.functional as F

from torch import nn
from einops import rearrange, repeat

from VIHE.models.utils.attn import (
    Conv2DBlock,
    Conv2DUpsampleBlock,
    PreNorm,
    Attention,
    RotaryAttention,
    cache_fn,
    DenseBlock,
    FeedForward,
)
from VIHE.models.utils.rotary_enc import RotaryEmbedding


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
        self.spatial_size = cfg.img_size // cfg.img_patch_size  # 128 / 8 = 16

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

        self.view_pe = nn.Parameter(
            torch.randn(
                self.num_img * 3,
                self.input_dim_before_seq,
            )
        )
        self.language_pe = nn.Parameter(
            torch.randn(
                1,
                self.lang_max_seq_len,
                cfg.attn_dim,
            )
        )

        if self.cfg.rotary_enc:
            self.img_rotray_emb = RotaryEmbedding(
                    dim = cfg.attn_dim // 2,
                    freqs_for = '3d',
                    max_freq = self.spatial_size,
                )
        else:
            self.patch_pe = nn.Parameter(
                torch.randn(
                    self.num_img * 3 * self.spatial_size**2,
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
            self.pixel_loc = self.pixel_loc.to(self.renderer.device)
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

        if self.cfg.rotary_enc:
            get_attn_attn = lambda: PreNorm(
                cfg.attn_dim,
                RotaryAttention(
                    cfg.attn_dim,
                    heads=cfg.attn_heads,
                    dim_head=cfg.attn_dim_head,
                    dropout=cfg.attn_dropout,
                ),
            )
        else:
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


        '''
        Cross Attention from action output to image language input
        '''
        if self.cfg.rotary_enc:
            get_cross_attn_attn = lambda: PreNorm(
                cfg.attn_dim,
                RotaryAttention(
                    cfg.attn_dim,
                    context_dim=cfg.attn_dim,
                    heads=cfg.attn_heads,
                    dim_head=cfg.attn_dim_head,
                    dropout=cfg.attn_dropout,
                ),
                context_dim=cfg.attn_dim,
            )
        else:
            get_cross_attn_attn = lambda: PreNorm(
                cfg.attn_dim,
                Attention(
                    cfg.attn_dim,
                    context_dim=cfg.attn_dim,
                    heads=cfg.attn_heads,
                    dim_head=cfg.attn_dim_head,
                    dropout=cfg.attn_dropout,
                ),
                context_dim=cfg.attn_dim,
            )

        get_cross_attn_ff = lambda: PreNorm(cfg.attn_dim, FeedForward(cfg.attn_dim))
        get_cross_attn_attn, get_cross_attn_ff = map(cache_fn, (get_cross_attn_attn, get_cross_attn_ff))
        

        cache_args = {"_cache": cfg.weight_tie_layers}

        self.cross_layers = nn.ModuleList([])
        for _ in range(cfg.depth):
            self.cross_layers.append(
                nn.ModuleList([get_cross_attn_attn(**cache_args), get_cross_attn_ff(**cache_args)])
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

    def get_pt_loc_on_img(self, pt, W_E_H_4x4, stage):
        """
        transform location of points in the local frame to location on the
        image
        :param pt: (bs, np, 3)
        :return: pt_img of size (bs, np, num_img, 2)
        """

        pt_img = self.renderer.get_pt_loc_on_img(
            pt, W_E_H_4x4=W_E_H_4x4, stage=stage
        )
        return pt_img

    def get_relative_loc_delta(self, view, bs, one_d_grid):
        grids = torch.zeros((bs, self.spatial_size, self.spatial_size, 3)).to(one_d_grid.device)

        if view == 'top':
            grids[:, :, :, 0] += one_d_grid[None, None, :]
            grids[:, :, :, 1] -= one_d_grid[None, :, None]
        elif view == 'front':
            grids[:, :, :, 0] -= one_d_grid[None, None, :]
            grids[:, :, :, 2] -= one_d_grid[None, :, None]
        elif view == 'back':
            grids[:, :, :, 0] += one_d_grid[None, None, :]
            grids[:, :, :, 2] -= one_d_grid[None, :, None]
        elif view == 'left':
            grids[:, :, :, 1] += one_d_grid[None, None, :]
            grids[:, :, :, 2] -= one_d_grid[None, :, None]
        elif view == 'right':
            grids[:, :, :, 1] -= one_d_grid[None, None, :]
            grids[:, :, :, 2] -= one_d_grid[None, :, None]
        else:
            assert False

        return grids

    def get_patch_loc_on_3d(self, stage, W_E_H_4x4):
        if self.cfg.zoom_in:
            if stage==0: max_range = 1
            elif stage==1: max_range = 0.5
            elif stage==2: max_range = 0.25
            else: assert False
        else:
            max_range = 1

        bs = W_E_H_4x4.shape[0]
        one_d_grid = torch.linspace(-max_range, max_range, steps=self.spatial_size).to(W_E_H_4x4.device)

        # top
        grid = torch.zeros((bs, self.num_img, self.spatial_size, self.spatial_size, 3)).to(W_E_H_4x4.device)
        for i, view in enumerate(['top', 'front', 'back', 'left', 'right']):
            delta_grid = self.get_relative_loc_delta(view, bs, one_d_grid)
            grid[:,i] = delta_grid

        if stage==0:
            grid[:,0,:,:,2] = 1 # top
            grid[:,1,:,:,1] = 1 # front 
            grid[:,2,:,:,1] = -1 # back
            grid[:,3,:,:,0] = 1 # left
            grid[:,4,:,:,0] = -1 # right
        else:
            # apply W_E_H_4x4
            grid_ = grid.view(bs,self.num_img*self.spatial_size**2, 3)
            grid_ = torch.cat((grid_, torch.ones((bs, self.num_img*self.spatial_size**2, 1)).to(W_E_H_4x4.device)), dim=-1)
            grid_ = torch.bmm(grid_, W_E_H_4x4.transpose(1, 2))[...,:3]
            grid = grid_.view(bs, self.num_img, self.spatial_size, self.spatial_size, 3)

        return grid


    def get_rotary_emb(self, patch_pos, stage=0):
        freq_0 = self.img_rotray_emb(patch_pos[..., 0])
        freq_1 = self.img_rotray_emb(patch_pos[..., 1])
        freq_2 = self.img_rotray_emb(patch_pos[..., 2])
        freq = torch.cat((freq_0, freq_1, freq_2), dim=-1)
        freq = rearrange(freq, "b ... d -> b (...) d") 

        bs = freq.shape[0]
        if self.cfg.add_lang and stage == 0:
            freq = torch.cat((self.language_pe.repeat(bs,1,1),freq), dim=1) # language first, image second

        return freq

    def forward_bef_attn(self, img, proprio, lang_emb, stage=0):
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

        # add view pe
        if self.cfg.rotary_enc:
            ins = ins + self.view_pe[stage*self.num_img:(stage+1)*self.num_img].unsqueeze(0).unsqueeze(2).unsqueeze(2)
            # flatten patches into sequence
            ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
        else:
            ins = rearrange(ins, "b ... d -> b (...) d")  # [B, num_img * np * np, 128]
            ins = ins + self.patch_pe[stage*self.num_img*self.spatial_size**2:(stage+1)*self.num_img*self.spatial_size**2].unsqueeze(0)

        # add learable pos encoding
        # only added to image tokens

        # append language features as sequence
        num_lang_tok = 0
        if self.cfg.add_lang and stage==0:
            l = self.lang_preprocess(
                lang_emb.view(bs * self.lang_max_seq_len, self.lang_emb_dim)
            )
            l = l.view(bs, self.lang_max_seq_len, -1)
            num_lang_tok = l.shape[1]
            ins = torch.cat((l, ins), dim=1)  # [B, num_img * np * np + 77, 128] # language first image second

        x = self.fc_bef_attn(ins)

        return x, d0, num_lang_tok

    def forward_after_attn(self, x, x_d0, stage):
        bs = x.shape[0]
        h, w = self.cfg.img_size, self.cfg.img_size
        x = self.fc_aft_attn(x)
        # reshape back to orginal size
        x = x.view(bs, self.num_img, self.spatial_size, self.spatial_size, x.shape[-1])  # [B, num_img, np, np, 128]
        x = rearrange(x, "b ... d -> b d ...")  # [B, 128, num_img, np, np]

        feat = []
        _feat = torch.max(torch.max(x, dim=-1)[0], dim=-1)[0]
        _feat = _feat.view(bs, -1)
        feat.append(_feat)

        x = (
            x.transpose(1, 2)
            .clone()
            .view(
                bs * self.num_img, self.input_dim_before_seq, self.spatial_size, self.spatial_size
            )
        )

        u0 = self.up0(x)
        u0 = torch.cat([u0, x_d0], dim=1)
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
        # import pdb; pdb.set_trace()

        out = {"trans": trans, "feat": feat}
        return out

    def forward(
        self,
        img,
        stage1_local_img,
        stage2_local_img,
        stage1_W_E_H_4x4, 
        stage2_W_E_H_4x4,
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

        if self.cfg.rotary_enc:
            global_0_pos = self.get_patch_loc_on_3d(0, stage1_W_E_H_4x4*0)
            local_1_pos = self.get_patch_loc_on_3d(1, stage1_W_E_H_4x4)
            local_2_pos = self.get_patch_loc_on_3d(2, stage2_W_E_H_4x4)

            global_freqs = self.get_rotary_emb(global_0_pos, stage=0)
            local_1_freqs = self.get_rotary_emb(local_1_pos, stage=1)
            local_2_freqs = self.get_rotary_emb(local_2_pos, stage=2)


 

        x, x_d0, num_lang_tok = self.forward_bef_attn(img, proprio, lang_emb, stage=0)
        stage1_local_x, stage1_local_x_d0, _ = self.forward_bef_attn(stage1_local_img, proprio, lang_emb, stage=1)
        stage2_local_x, stage2_local_x_d0, _ = self.forward_bef_attn(stage2_local_img, proprio, lang_emb, stage=2)
        cross_attn_ratio_list = []

        for i in range(self.cfg.depth):
            cross_attn, cross_ff = self.cross_layers[i]
            # cross attention
            if self.cfg.rotary_enc:
                if self.cfg.cross_stage:
                    stage2_local_x_, stage2_local_attn_ = cross_attn(
                            stage2_local_x, 
                            x_freq=local_2_freqs, 
                            context=torch.cat((x, stage1_local_x, stage2_local_x), dim=1),
                            context_freq=torch.cat((global_freqs, local_1_freqs, local_2_freqs), dim=1)
                            )
                    
                    stage1_local_x_, stage1_local_attn_ = cross_attn(
                            stage1_local_x, 
                            x_freq=local_1_freqs, 
                            context=torch.cat((x, stage1_local_x), dim=1),
                            context_freq=torch.cat((global_freqs, local_1_freqs), dim=1)
                            )
                    x_, _ = cross_attn(
                            x, 
                            x_freq=global_freqs
                            )
                    
                else:
                    stage2_local_x_, _ = cross_attn(
                            stage2_local_x, 
                            x_freq=local_2_freqs, 
                            )
                    stage1_local_x_, _  = cross_attn(
                            stage1_local_x, 
                            x_freq=local_1_freqs, 
                            )
                    x_, _ = cross_attn(
                            x, 
                            x_freq=global_freqs
                            )             
            else:
                if self.cfg.cross_stage:
                    stage2_local_x_ = cross_attn(
                            stage2_local_x, 
                            context=torch.cat((x, stage1_local_x, stage2_local_x), dim=1),
                            )
                    stage1_local_x_ = cross_attn(
                            stage1_local_x, 
                            context=torch.cat((x, stage1_local_x), dim=1),
                            )
                    x_ = cross_attn(
                            x, 
                            )
                else:
                    raise NotImplementedError


            stage2_local_x = stage2_local_x + stage2_local_x_
            stage1_local_x = stage1_local_x + stage1_local_x_
            x = x + x_

            # ffn
            stage2_local_x = cross_ff(stage2_local_x) + stage2_local_x
            stage1_local_x = cross_ff(stage1_local_x) + stage1_local_x
            x = cross_ff(x) + x

        # append language features as sequence
        if self.cfg.add_lang:
            # throwing away the language embeddings
            x = x[:, num_lang_tok:]


        out = self.forward_after_attn(x, x_d0, stage=0)
        stage1_local_out = self.forward_after_attn(stage1_local_x, stage1_local_x_d0, stage=1)
        stage2_local_out = self.forward_after_attn(stage2_local_x, stage2_local_x_d0, stage=2)

        out = {
            'out': out,
            'stage1_local_out': stage1_local_out,
            'stage2_local_out': stage2_local_out,
        }

        return out

    def get_wpt(self, out, W_E_H_4x4, stage):
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

        fix_cam = stage==0
        pred_wpt = [
            self.renderer.get_max_3d_frm_hm_cube(
                hm[i : i + 1],
                stage=stage,
                fix_cam=fix_cam,
                W_E_H_4x4=W_E_H_4x4[i : i + 1],
            )
            for i in range(bs)
        ]
        pred_wpt = torch.cat(pred_wpt, 0)

        return pred_wpt

    def free_mem(self):
        """
        Could be used for freeing up the memory once a batch of testing is done
        """
        print("Freeing up some memory")
        self.renderer.free_mem()
