"""Architecture, checkpoint-compatibility and attribution tests. Random initialisation only (no downloads)."""
import numpy as np
import pytest
import torch

from omari_us import explainability as X
from omari_us.models import build_model
from omari_us.models.usf_mae import (build_mae_from_state_dict, get_2d_sincos_pos_embed, load_usf_mae_checkpoint, mae_to_vit_classifier,
                                     mae_vit_base_patch16_dec512d8b, reconstruct)

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def densenet():
    torch.manual_seed(0)
    return build_model("densenet161", pretrained=False).eval()


@pytest.fixture(scope="module")
def vit():
    torch.manual_seed(0)
    return build_model("vit_b16", pretrained=False).eval()


def test_param_counts_match_website(densenet, vit):
    # the website (src/lib/architectures.ts) computes these analytically; tests on both sides pin the same numbers
    assert sum(p.numel() for p in densenet.parameters()) == 26_485_254
    assert sum(p.numel() for p in vit.parameters()) == 85_803_270


def test_sincos_matches_reference_values():
    e = get_2d_sincos_pos_embed(768, 14, cls_token=True)
    assert e.shape == (197, 768) and np.allclose(e[0], 0)
    assert np.allclose([e[16][i] for i in (0, 1, 383, 384, 385, 767)], [0.841471, 0.815251, 1.0, 0.841471, 0.815251, 1.0], atol=1e-5)


def test_mae_checkpoint_roundtrip_and_encoder_transfer(tmp_path):
    torch.manual_seed(1)
    mae = mae_vit_base_patch16_dec512d8b().eval()
    # save in the container format used by MAE / USF-MAE training scripts, with a DataParallel prefix
    path = tmp_path / "usf.pth"
    torch.save({"model": {"module." + k: v for k, v in mae.state_dict().items()}, "epoch": 99}, path)
    sd, rep = load_usf_mae_checkpoint(path)
    assert rep.ok_for_classification and rep.ok_for_reconstruction and rep.pos_embed_is_sincos
    vit = mae_to_vit_classifier(sd, num_classes=6, global_pool="token", drop_path_rate=0.0).eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        # MAE encoder with no masking (identity noise order keeps token order) must equal the timm ViT features
        noise = torch.arange(196, dtype=torch.float32)[None]
        lat, mask, _ = mae.forward_encoder(x, 0.0, noise)
        feats = vit.forward_features(x)
    assert mask.sum() == 0
    assert torch.allclose(lat, feats, atol=1e-5)


def test_encoder_only_checkpoint_rejected_for_reconstruction(tmp_path):
    mae = mae_vit_base_patch16_dec512d8b()
    enc = {k: v for k, v in mae.state_dict().items() if not k.startswith(("decoder", "mask_token"))}
    torch.save(enc, tmp_path / "enc.pth")
    sd, rep = load_usf_mae_checkpoint(tmp_path / "enc.pth")
    assert rep.ok_for_classification and not rep.ok_for_reconstruction
    with pytest.raises(ValueError):
        build_mae_from_state_dict(sd)


def test_reconstruct_shapes_and_mask_count():
    torch.manual_seed(2)
    mae = mae_vit_base_patch16_dec512d8b().eval()
    r = reconstruct(mae, torch.randn(1, 3, 224, 224), 0.75, seed=3)
    assert r["pred_img"].shape == (1, 3, 224, 224) and len(r["masked_indices"][0]) == 147
    r2 = reconstruct(mae, torch.randn(1, 3, 224, 224), 0.75, seed=3)
    assert r["masked_indices"] == r2["masked_indices"]  # seeded masks are reproducible


def test_gradcam_equals_relu_cam_over_z_for_densenet(densenet):
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    c = 2
    cam = X.explain("cam", "densenet161", densenet, x, c)
    gc = X.explain("gradcam", "densenet161", densenet, x, c)
    assert cam.shape == (7, 7)
    assert torch.allclose(gc, torch.relu(cam) / 49, atol=1e-5)


def test_gradcampp_and_layers(densenet):
    x = torch.randn(1, 3, 224, 224)
    m = X.explain("gradcampp", "densenet161", densenet, x, 1)
    assert m.shape == (7, 7) and (m >= 0).all()
    m3 = X.explain("gradcam", "densenet161", densenet, x, 1, "features.denseblock3")
    assert m3.shape == (14, 14)


def test_vit_methods(vit):
    x = torch.randn(1, 3, 224, 224)
    for meth in ("vit_gradcam", "vit_gradcampp", "attention_rollout", "grad_rollout"):
        m = X.explain(meth, "vit_b16", vit, x, 3)
        assert m.shape == (14, 14) and torch.isfinite(m).all(), meth
    before = [getattr(b.attn, "fused_attn", False) for b in vit.blocks]
    A = X.vitx.attention_maps(vit, x)
    assert [getattr(b.attn, "fused_attn", False) for b in vit.blocks] == before  # fused-attention flag restored
    assert A.shape == (12, 12, 197, 197)
    assert torch.allclose(A.sum(-1), torch.ones(12, 12, 197), atol=1e-4)


def test_grad_rollout_rename_and_legacy_alias(vit):
    # schema 1.1: "chefer" was renamed "grad_rollout"; the old id still resolves (old bundles) but is not listed
    assert "grad_rollout" in X.methods_for("vit_b16") and "chefer" not in X.methods_for("vit_b16")
    assert X.canonical_method("chefer") == "grad_rollout" and X.compatible("chefer", "usf_mae")
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(X.explain("chefer", "vit_b16", vit, x, 1), X.explain("grad_rollout", "vit_b16", vit, x, 1))
    assert X.method_label("chefer") == X.method_label("grad_rollout") == "Gradient-weighted attention rollout"


def test_layer_labels_are_one_indexed_after_block():
    # blockN = input of model.blocks[N] = residual stream after N blocks -> "after block N of 12"
    assert X.layer_label("vit_b16", "block11", "vit_gradcam") == "after block 11 of 12"
    assert X.layer_label("usf_mae", "block5", "vit_gradcampp") == "after block 5 of 12"
    assert X.layer_label("vit_b16", "final", "attention_rollout") == "all 12 blocks (rollout)"
    assert X.layer_label("densenet161", "features.denseblock3", "gradcam") == "dense block 3 output (14x14)"
    assert X.layer_label("densenet161", "act", "cam").startswith("final features")


def test_incompatible_and_reserved(vit, densenet):
    x = torch.randn(1, 3, 224, 224)
    with pytest.raises(X.IncompatibleMethod):
        X.explain("cam", "vit_b16", vit, x, 0)
    with pytest.raises(NotImplementedError):
        X.explain("metacampp", "densenet161", densenet, x, 0)
    assert "metacampp" not in X.methods_for("densenet161")


def test_deletion_curve_monotone_endpoints(densenet):
    x = torch.randn(1, 3, 224, 224)
    m = X.explain("gradcam", "densenet161", densenet, x, 0)
    r = X.faithfulness.deletion_insertion(densenet, x, m, 0, steps=4)
    assert len(r["curve"]) == 5 and 0 <= r["auc"] <= 1
