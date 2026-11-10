import numpy as np
import pytest

from omari_us.datasets import check_patient_disjoint, class_weights, dataset_stats, find_near_duplicates, label_fraction_subset, make_splits, synthetic_metadata
from omari_us.datasets import perturbations as P
from omari_us.datasets.fetal_planes import plane_to_label, read_metadata


def test_plane_aliases():
    assert plane_to_label(" Fetal brain ") == 1 and plane_to_label("Maternal cervix") == 4 and plane_to_label("Other") == 5
    with pytest.raises(ValueError):
        plane_to_label("heart")


def test_read_metadata_semicolon_and_trailing_spaces(tmp_path):
    img = tmp_path / "Images"
    df0 = synthetic_metadata(6, (1, 3), image_dir=str(img))
    csv = tmp_path / "FETAL_PLANES_DB_data.csv"
    out = df0[["Image_name", "Patient_num", "Plane", "Brain_plane", "Operator", "US_Machine", "Train"]].rename(columns={"Train": "Train "})
    out.to_csv(csv, sep=";", index=False)
    df = read_metadata(tmp_path)
    assert len(df) == len(df0) and "Train" in df and df["path"].str.endswith(".png").all()


def test_splits_patient_disjoint_and_keep_authors_test():
    df = make_splits(synthetic_metadata(80), 0.1, 0)
    counts = check_patient_disjoint(df)
    assert set(counts) == {"train", "val", "test"}
    assert (df.loc[df.split == "test", "Train"] == 0).all()
    assert (df.loc[df.split != "test", "Train"] == 1).all()


def test_label_fraction_subset_by_patient():
    df = make_splits(synthetic_metadata(200, (2, 6), seed=3), 0.1, 0)
    sub = label_fraction_subset(df, 0.1, seed=0)
    full = df[df.split == "train"]
    pats = sub.Patient_num.unique()
    # whole patients are kept
    assert all((full.Patient_num == p).sum() == (sub.Patient_num == p).sum() for p in pats)
    assert sub.Patient_num.nunique() < 0.25 * full.Patient_num.nunique()
    assert set(sub.label) == set(full.label)


def test_class_weights_inverse():
    w = class_weights([0, 0, 0, 1], "inverse")
    assert w[1] > w[0] and abs(w.mean() - 1) < 1e-9


def test_near_duplicates_and_stats(tmp_path):
    df = make_splits(synthetic_metadata(40, (2, 3), image_dir=str(tmp_path)), 0.2, 0)
    # copy one image to create an exact duplicate in another row
    import shutil

    shutil.copy(df.path.iloc[0], df.path.iloc[-1])
    d = find_near_duplicates(df, 0, progress=False)
    assert ((d.a == 0) & (d.b == len(df) - 1)).any()
    s = dataset_stats(df, d, 0)
    assert s["total_images"] == len(df) and sum(s["images_per_patient"]) == len(df)
    assert s["near_duplicates"]["images_in_groups"] >= 2


def test_perturbation_semantics_match_website():
    img = np.full((8, 8), 0.4, np.float32)
    assert np.allclose(P.contrast(img, 2.0), 0.3)  # (0.4-0.5)*2+0.5
    assert np.allclose(P.brightness(img, 0.15), 0.55)
    big = np.full((200, 200), 0.4, np.float32)
    s = P.speckle(big, 1.0, looks=4, seed=0) / 0.4
    assert abs(s.mean() - 1) < 0.02 and abs(s.var() - 0.25) < 0.03
