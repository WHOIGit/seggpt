import torch
import torch.nn as nn
import torch.nn.functional as F

##########################
from fairscale.nn.checkpoint import checkpoint_wrapper
from timm.models.layers import trunc_normal_
from .backend import Block

from .util.vitdet_utils import (
    PatchEmbed,
    get_abs_pos,
    LayerNorm2D,
)

class SegGPT(nn.Module):
    def __init__(
             self,
             img_size=224,
             patch_size=16,
             in_chans=3,
             embed_dim=1024,
             depth=24,
             num_heads=16,
             mlp_ratio=4.,
             qkv_bias=True,
             drop_path_rate=0.,
             norm_layer=nn.LayerNorm,
             act_layer=nn.GELU,
             use_abs_pos=True,
             use_rel_pos=False,
             rel_pos_zero_init=True,
             window_size=0,
             window_block_indexes=(),
             residual_block_indexes=(),
             use_act_checkpoint=False,
             pretrain_img_size=224,
             pretrain_use_cls_token=True,
             out_feature="last_feat",
             decoder_embed_dim=128,
             loss_func="smoothl1",
             ):
        super().__init__()

        # --------------------------------------------------------------------------
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_x = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.segment_token_y = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # token for seg types
        self.type_token_cls = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        self.type_token_ins = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size[0] // patch_size, img_size[1] // patch_size),
            )
            if use_act_checkpoint:
                block = checkpoint_wrapper(block)
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim*4, patch_size ** 2 * self.decoder_embed_dim, bias=True)  # decoder to patch
        self.decoder_pred = nn.Sequential(
                nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1, ),
                LayerNorm2D(self.decoder_embed_dim),
                nn.GELU(),
                nn.Conv2d(self.decoder_embed_dim, 3, kernel_size=1, bias=True), # decoder to patch
        )
        # --------------------------------------------------------------------------
        self.loss_func = loss_func

        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.segment_token_x, std=.02)
        torch.nn.init.normal_(self.segment_token_y, std=.02)
        torch.nn.init.normal_(self.type_token_cls, std=.02)
        torch.nn.init.normal_(self.type_token_ins, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == 2 * imgs.shape[3] and imgs.shape[2] % p == 0

        w = imgs.shape[3] // p
        h = w * 2
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        w = int((x.shape[1]*0.5)**.5)
        h = w * 2
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def forward_encoder(self, imgs, tgts, bool_masked_pos, seg_type, merge_between_batch=-1):
        # embed patches
        x = self.patch_embed(imgs)
        y = self.patch_embed(tgts)
        batch_size, Hp, Wp, _ = x.size()
        seq_len = Hp * Wp

        mask_token = self.mask_token.expand(batch_size, Hp, Wp, -1)
        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token).reshape(-1, Hp, Wp, 1)
        y = y * (1 - w) + mask_token * w

        # add pos embed w/o cls token
        x = x + self.segment_token_x
        y = y + self.segment_token_y
        if self.pos_embed is not None:
            x = x + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
            )
            y = y + get_abs_pos(
                self.pos_embed, self.pretrain_use_cls_token, (y.shape[1], y.shape[2])
            )

        # add type tokens for cls and ins
        type_emb = torch.zeros(batch_size, 1, 1, self.type_token_cls.shape[-1]).to(x.device)
        type_emb[seg_type==0] = self.type_token_cls
        type_emb[seg_type==1] = self.type_token_ins

        x = x + type_emb
        y = y + type_emb
        x = torch.cat((x, y), dim=0)
        merge_idx = 2
        # apply Transformer blocks
        out = []
        for idx, blk in enumerate(self.blocks):
            merge = 0
            if merge_between_batch >= 0 and idx >= merge_between_batch:
                merge = 1 if merge_idx >= idx else 2
            x = blk(x, merge=merge)
            if idx == merge_idx:
                x = (x[:x.shape[0]//2] + x[x.shape[0]//2:]) * 0.5
            if idx in [5, 11, 17, 23]:
                out.append(self.norm(x))
        return out

    def forward_decoder(self, x):
        x = torch.cat(x, dim=-1)
        x = self.decoder_embed(x) # BxhxwxC
        p = self.patch_size
        h, w = x.shape[1], x.shape[2]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.decoder_embed_dim))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, h * p, w * p))

        x = self.decoder_pred(x) # Bx3xHxW
        return x

    def forward_loss(self, pred, tgts, mask, valid):
        """
        tgts: [N, 3, H, W]
        pred: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove, 
        valid: [N, 3, H, W]
        """
        mask = mask[:, :, None].repeat(1, 1, self.patch_size**2 * 3)
        mask = self.unpatchify(mask)
        mask = mask * valid

        target = tgts
        if self.loss_func == "l1l2":
            loss = ((pred - target).abs() + (pred - target) ** 2.) * 0.5
        elif self.loss_func == "l1":
            loss = (pred - target).abs()
        elif self.loss_func == "l2":
            loss = (pred - target) ** 2.
        elif self.loss_func == "smoothl1":
            loss = F.smooth_l1_loss(pred, target, reduction="none", beta=0.01)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, tgts, bool_masked_pos=None, valid=None, seg_type=None, merge_between_batch=-1):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((imgs.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(imgs.device)
        else:
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        latent = self.forward_encoder(imgs, tgts, bool_masked_pos, seg_type, merge_between_batch=merge_between_batch)
        pred = self.forward_decoder(latent)  # [N, L, p*p*3]
        loss = self.forward_loss(pred, tgts, bool_masked_pos, valid)
        return loss, self.patchify(pred), bool_masked_pos
