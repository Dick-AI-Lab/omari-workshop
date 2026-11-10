"""USF-MAE: a state-dict-compatible re-implementation of `mae_vit_base_patch16_dec512d8b` plus checkpoint tools.

USF-MAE (Megahed et al., BSPC 2026; https://github.com/Yusufii9/USF-MAE) pretrains Facebook's MAE
(He et al., CVPR 2022) on ≈370k ultrasound images. Its public pretraining notebook builds
`mae.models_mae.mae_vit_base_patch16_dec512d8b()` and first loads `mae_pretrain_vit_base_full.pth`
(ImageNet-1k MAE). The 100-epoch checkpoint is distributed via Google Drive (see the repository README).

This module re-implements the MAE model with *identical parameter names and shapes* using timm building
blocks, so that the official checkpoint loads with `strict=True` without vendoring facebookresearch/mae
(CC BY-NC 4.0). Positional embeddings are the fixed 2D sin-cos tables from MAE's util/pos_embed.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

USF_MAE_REPO = "https://github.com/Yusufii9/USF-MAE"
USF_MAE_100EP_GDRIVE = "https://drive.google.com/file/d/1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr/view"
USF_MAE_100EP_GDRIVE_ID = "1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr"
USF_MAE_500EP_GDRIVE_ID = "15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg"


# ---------------------------------------------------------------- positional embeddings (MAE util/pos_embed.py)
def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    pos = np.concatenate([emb_h, emb_w], axis=1)
    if cls_token:
        pos = np.concatenate([np.zeros([1, embed_dim]), pos], axis=0)
    return pos


# ---------------------------------------------------------------- model
class MaskedAutoencoderViT(nn.Module):
    """Parameter names/shapes identical to facebookresearch/mae models_mae.MaskedAutoencoderViT."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_ratio=4.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False):
        super().__init__()
        from timm.models.vision_transformer import Block, PatchEmbed

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.initialize_weights()

    def initialize_weights(self):
        g = int(self.patch_embed.num_patches ** 0.5)
        self.pos_embed.data.copy_(torch.from_numpy(get_2d_sincos_pos_embed(self.pos_embed.shape[-1], g, cls_token=True)).float().unsqueeze(0))
        self.decoder_pos_embed.data.copy_(torch.from_numpy(get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], g, cls_token=True)).float().unsqueeze(0))
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # -- patch utilities (same conventions as MAE)
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_chans, h, p, w, p)
        x = torch.einsum("nchpwq->nhwpqc", x)
        return x.reshape(imgs.shape[0], h * w, p ** 2 * self.in_chans)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, self.in_chans)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], self.in_chans, h * p, h * p)

    def random_masking(self, x: torch.Tensor, mask_ratio: float, noise: Optional[torch.Tensor] = None):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        if noise is None:
            noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, noise=None):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio, noise)
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x), mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        loss = ((pred - target) ** 2).mean(dim=-1)
        return (loss * mask).sum() / mask.sum().clamp(min=1)

    def forward(self, imgs, mask_ratio: float = 0.75, noise: Optional[torch.Tensor] = None):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, noise)
        pred = self.forward_decoder(latent, ids_restore)
        return self.forward_loss(imgs, pred, mask), pred, mask


def mae_vit_base_patch16_dec512d8b(**kw) -> MaskedAutoencoderViT:
    return MaskedAutoencoderViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=512, decoder_depth=8,
                                decoder_num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kw)


# ---------------------------------------------------------------- checkpoint handling
ENCODER_PREFIXES = ("cls_token", "pos_embed", "patch_embed.", "blocks.", "norm.")
DECODER_PREFIXES = ("decoder_embed.", "mask_token", "decoder_pos_embed", "decoder_blocks.", "decoder_norm.", "decoder_pred.")


@dataclass
class CheckpointReport:
    path: str
    sha256: str
    container_keys: List[str]
    n_tensors: int
    has_encoder: bool
    has_decoder: bool
    missing_encoder: List[str] = field(default_factory=list)
    missing_decoder: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    shape_mismatches: List[Tuple[str, Tuple, Tuple]] = field(default_factory=list)
    pos_embed_is_sincos: Optional[bool] = None

    @property
    def ok_for_classification(self) -> bool:
        return self.has_encoder and not self.missing_encoder and not [m for m in self.shape_mismatches if m[0].startswith(ENCODER_PREFIXES)]

    @property
    def ok_for_reconstruction(self) -> bool:
        return self.ok_for_classification and self.has_decoder and not self.missing_decoder

    def summary(self) -> str:
        lines = [f"checkpoint: {self.path}", f"sha256: {self.sha256}", f"container keys: {self.container_keys}", f"tensors: {self.n_tensors}",
                 f"encoder complete: {self.ok_for_classification}", f"decoder present: {self.has_decoder} (complete: {self.ok_for_reconstruction})",
                 f"pos_embed equals MAE sin-cos table: {self.pos_embed_is_sincos}"]
        if self.missing_encoder:
            lines.append(f"missing encoder keys ({len(self.missing_encoder)}): {self.missing_encoder[:6]}")
        if self.missing_decoder and self.has_decoder:
            lines.append(f"missing decoder keys ({len(self.missing_decoder)}): {self.missing_decoder[:6]}")
        if self.unexpected:
            lines.append(f"unexpected keys ({len(self.unexpected)}): {self.unexpected[:6]}")
        if self.shape_mismatches:
            lines.append(f"shape mismatches: {self.shape_mismatches[:6]}")
        return "\n".join(lines)


def _extract_state_dict(obj) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    container = list(obj.keys()) if isinstance(obj, dict) else []
    sd = obj
    for k in ("model", "state_dict", "model_state_dict", "encoder"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint does not contain a state dict")
    clean = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        for pre in ("module.", "_orig_mod.", "model."):
            if k.startswith(pre):
                k = k[len(pre):]
        clean[k] = v
    return clean, container


def load_usf_mae_checkpoint(path: str | os.PathLike, map_location="cpu") -> Tuple[Dict[str, torch.Tensor], CheckpointReport]:
    """Load a USF-MAE / MAE checkpoint and verify it against `mae_vit_base_patch16_dec512d8b`.

    Returns (state_dict, report). Raises if the encoder is unusable.
    """
    from ..utils import sha256_file

    obj = torch.load(path, map_location=map_location, weights_only=False)
    sd, container = _extract_state_dict(obj)
    ref = mae_vit_base_patch16_dec512d8b().state_dict()
    enc_keys = [k for k in ref if k.startswith(ENCODER_PREFIXES)]
    dec_keys = [k for k in ref if k.startswith(DECODER_PREFIXES)]
    rep = CheckpointReport(
        path=str(path), sha256=sha256_file(path), container_keys=container, n_tensors=len(sd),
        has_encoder=any(k.startswith("blocks.") for k in sd), has_decoder=any(k.startswith("decoder_") for k in sd),
        missing_encoder=[k for k in enc_keys if k not in sd], missing_decoder=[k for k in dec_keys if k not in sd],
        unexpected=[k for k in sd if k not in ref],
        shape_mismatches=[(k, tuple(sd[k].shape), tuple(ref[k].shape)) for k in sd if k in ref and sd[k].shape != ref[k].shape],
    )
    if "pos_embed" in sd and sd["pos_embed"].shape == ref["pos_embed"].shape:
        rep.pos_embed_is_sincos = bool(torch.allclose(sd["pos_embed"].float(), ref["pos_embed"].float(), atol=1e-5))
    if not rep.ok_for_classification:
        raise ValueError("Checkpoint is not a usable MAE ViT-B/16 encoder:\n" + rep.summary())
    return sd, rep


def build_mae_from_state_dict(sd: Dict[str, torch.Tensor]) -> MaskedAutoencoderViT:
    """Full MAE (encoder + decoder) for reconstruction; requires decoder weights."""
    m = mae_vit_base_patch16_dec512d8b()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k.startswith(DECODER_PREFIXES + ENCODER_PREFIXES)]
    if missing:
        raise ValueError(f"Cannot build reconstruction model; missing {len(missing)} keys, e.g. {missing[:5]}")
    return m.eval()


def mae_to_vit_classifier(sd: Dict[str, torch.Tensor], num_classes: int = 6, global_pool: str = "token", drop_path_rate: float = 0.1):
    """Create a timm VisionTransformer (ViT-B/16) and load the MAE *encoder* weights into it.

    global_pool='token': classify from [CLS] after the pretrained final LayerNorm (same head as supervised ViT-B/16,
                         keeps attribution code identical across the two ViTs).
    global_pool='avg':   MAE fine-tuning recipe — mean of patch tokens + new fc_norm; pretrained `norm` is unused.
    """
    import timm

    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes, global_pool=global_pool, drop_path_rate=drop_path_rate)
    enc = {k: v for k, v in sd.items() if k.startswith(ENCODER_PREFIXES)}
    if global_pool == "avg":
        enc = {k: v for k, v in enc.items() if not k.startswith("norm.")}
    missing, unexpected = model.load_state_dict(enc, strict=False)
    allowed_missing = ("head.", "fc_norm.") if global_pool == "avg" else ("head.",)
    bad_missing = [k for k in missing if not k.startswith(allowed_missing)]
    if bad_missing or unexpected:
        raise RuntimeError(f"Encoder transfer failed. missing={bad_missing[:6]} unexpected={list(unexpected)[:6]}")
    nn.init.trunc_normal_(model.head.weight, std=2e-5)  # MAE fine-tuning head init
    nn.init.zeros_(model.head.bias)
    return model


@torch.no_grad()
def reconstruct(mae: MaskedAutoencoderViT, x: torch.Tensor, mask_ratio: float, seed: int = 0):
    """Reconstruction for one normalised image batch x (B,3,224,224) with a seeded mask.

    Returns dict with: pred_img (B,3,H,W) normalised space, pasted (visible from input + predicted masked),
    mask (B, L) 1 = masked, loss (masked MSE, scalar), masked_indices (list per image).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.rand(x.shape[0], mae.patch_embed.num_patches, generator=g).to(x.device)
    loss, pred, mask = mae(x, mask_ratio=mask_ratio, noise=noise)
    pred_img = mae.unpatchify(pred)
    m = mask.unsqueeze(-1).repeat(1, 1, mae.patch_size ** 2 * mae.in_chans)
    m = mae.unpatchify(m)
    pasted = x * (1 - m) + pred_img * m
    idx = [torch.nonzero(mask[i] > 0.5).flatten().tolist() for i in range(mask.shape[0])]
    return {"pred_img": pred_img, "pasted": pasted, "mask": mask, "loss": float(loss), "masked_indices": idx}


def download_checkpoint_from_gdrive(file_id: str = USF_MAE_100EP_GDRIVE_ID, out: str = "checkpoints/usf_mae_100ep.pth") -> str:
    """Download via `gdown` (pip install gdown). Large Google Drive files require the confirmation flow gdown handles."""
    import gdown  # type: ignore

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    gdown.download(id=file_id, output=out, quiet=False)
    return out
