"""FETAL_PLANES_DB: metadata loading, patient-level splitting, label-fraction subsets, statistics, duplicates.

Dataset: Burgos-Artizzu et al., Sci. Rep. 10:10200 (2020); Zenodo record 3904280; CC BY 4.0.
Metadata file: FETAL_PLANES_DB_data.csv (semicolon-separated) with columns
    Image_name; Patient_num; Plane; Brain_plane; Operator; US_Machine; Train
`Train` is 1 for the authors' training split and 0 for their test split. Column names may carry trailing
spaces in the released CSV ("Train "), so they are stripped here.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils import CLASSES, NUM_CLASSES, now_iso

PLANE_ALIASES = {
    "fetal abdomen": 0, "abdomen": 0,
    "fetal brain": 1, "brain": 1,
    "fetal femur": 2, "femur": 2,
    "fetal thorax": 3, "thorax": 3,
    "maternal cervix": 4, "cervix": 4,
    "other": 5,
}

ZENODO_URL = "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
ZENODO_MD5 = "2a5fcc2cefb789bcc0f6c1f73e0ea43f"


def plane_to_label(plane: str) -> int:
    key = " ".join(str(plane).strip().lower().split())
    if key not in PLANE_ALIASES:
        raise ValueError(f"Unknown plane label {plane!r}; expected one of {CLASSES}")
    return PLANE_ALIASES[key]


def find_metadata_csv(root: str | os.PathLike) -> Path:
    root = Path(root)
    cands = sorted(glob.glob(str(root / "**" / "FETAL_PLANES_DB_data.csv"), recursive=True))
    if not cands:
        cands = sorted(glob.glob(str(root / "**" / "*.csv"), recursive=True))
    if not cands:
        raise FileNotFoundError(f"No metadata CSV found under {root}. Extract FETAL_PLANES_ZENODO.zip there first.")
    return Path(cands[0])


def find_image_dir(root: str | os.PathLike) -> Path:
    root = Path(root)
    for c in [root / "Images", *[Path(p) for p in glob.glob(str(root / "**" / "Images"), recursive=True)]]:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"No 'Images' directory found under {root}.")


def read_metadata(root_or_csv: str | os.PathLike, image_dir: Optional[str | os.PathLike] = None, check_files: bool = True) -> pd.DataFrame:
    """Load and validate the metadata CSV. Returns one row per image with integer `label` and absolute `path`."""
    p = Path(root_or_csv)
    csv = p if p.suffix.lower() == ".csv" else find_metadata_csv(p)
    df = pd.read_csv(csv, sep=";")
    if df.shape[1] == 1:  # fall back to comma separation
        df = pd.read_csv(csv)
    df.columns = [c.strip() for c in df.columns]
    required = {"Image_name", "Patient_num", "Plane", "Train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata CSV {csv} is missing columns {sorted(missing)}; found {list(df.columns)}")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    df["label"] = df["Plane"].map(plane_to_label)
    df["Train"] = df["Train"].astype(int)
    df["Patient_num"] = df["Patient_num"].astype(int)
    img_dir = Path(image_dir) if image_dir else (csv.parent / "Images" if (csv.parent / "Images").is_dir() else None)
    if img_dir is None:
        try:
            img_dir = find_image_dir(csv.parent)
        except FileNotFoundError:
            img_dir = csv.parent
    df["path"] = [str(img_dir / (n if n.lower().endswith(".png") else n + ".png")) for n in df["Image_name"]]
    if check_files:
        exists = df["path"].map(os.path.exists)
        if not exists.all():
            n = int((~exists).sum())
            raise FileNotFoundError(f"{n} image files listed in {csv} were not found in {img_dir} (first: {df.loc[~exists, 'path'].iloc[0]})")
    if df["Image_name"].duplicated().any():
        raise ValueError("Duplicate Image_name entries in metadata")
    return df.reset_index(drop=True)


def make_splits(df: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Keep the authors' train/test split; carve a patient-disjoint, class-stratified validation set from training.

    Adds a `split` column with values 'train' | 'val' | 'test'.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    out = df.copy()
    out["split"] = np.where(out["Train"] == 1, "train", "test")
    tr = out[out["split"] == "train"]
    n_splits = max(2, int(round(1 / val_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, va_idx = next(sgkf.split(tr, y=tr["label"], groups=tr["Patient_num"]))
    out.loc[tr.index[va_idx], "split"] = "val"
    check_patient_disjoint(out)
    return out


def check_patient_disjoint(df: pd.DataFrame) -> Dict[str, int]:
    """Raise if any patient appears in more than one split. Returns patient counts per split."""
    groups = {s: set(df.loc[df["split"] == s, "Patient_num"]) for s in df["split"].unique()}
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = groups[names[i]] & groups[names[j]]
            if inter:
                raise AssertionError(f"Patient leakage between {names[i]} and {names[j]}: {len(inter)} patients, e.g. {sorted(inter)[:5]}")
    return {k: len(v) for k, v in groups.items()}


def label_fraction_subset(df: pd.DataFrame, fraction: float, seed: int = 0, split: str = "train") -> pd.DataFrame:
    """Sample a fraction of *patients* from `split` (stratified by each patient's majority class), keeping all their images.

    Sampling by patient keeps the subset free of within-patient duplicates of held-out patients and mimics
    annotating fewer studies. Guarantees at least one patient per class when possible.
    """
    if fraction >= 1.0:
        return df[df["split"] == split].copy()
    sub = df[df["split"] == split]
    rng = np.random.default_rng(seed)
    maj = sub.groupby("Patient_num")["label"].agg(lambda s: s.value_counts().idxmax())
    chosen: List[int] = []
    for c in range(NUM_CLASSES):
        pats = maj[maj == c].index.to_numpy()
        if len(pats) == 0:
            continue
        k = max(1, int(round(fraction * len(pats))))
        chosen.extend(rng.choice(pats, size=min(k, len(pats)), replace=False).tolist())
    out = sub[sub["Patient_num"].isin(chosen)].copy()
    # ensure every class has at least one image
    for c in range(NUM_CLASSES):
        if not (out["label"] == c).any() and (sub["label"] == c).any():
            p = rng.choice(sub.loc[sub["label"] == c, "Patient_num"].unique())
            out = pd.concat([out, sub[sub["Patient_num"] == p]])
    return out.drop_duplicates("Image_name")


def class_weights(labels: Sequence[int], scheme: str = "inverse", beta: float = 0.999) -> np.ndarray:
    """'none' | 'inverse' (N / (K n_c)) | 'sqrt_inverse' | 'effective' (Cui et al. 2019). Normalised to mean 1."""
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES).astype(float)
    counts[counts == 0] = 1.0
    if scheme == "none":
        w = np.ones(NUM_CLASSES)
    elif scheme == "inverse":
        w = counts.sum() / (NUM_CLASSES * counts)
    elif scheme == "sqrt_inverse":
        w = np.sqrt(counts.sum() / (NUM_CLASSES * counts))
    elif scheme == "effective":
        w = (1 - beta) / (1 - np.power(beta, counts))
    else:
        raise ValueError(scheme)
    return w * NUM_CLASSES / w.sum()


# ----------------------------------------------------------------------------------------------
# duplicates
# ----------------------------------------------------------------------------------------------

def dhash(path: str, size: int = 8) -> int:
    """Difference hash (64-bit for size 8) of a grayscale thumbnail; robust to small intensity changes."""
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.BILINEAR)
        a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def find_near_duplicates(df: pd.DataFrame, max_hamming: int = 4, hashes: Optional[np.ndarray] = None, progress: bool = True) -> pd.DataFrame:
    """Pairs of images whose dHash differs by ≤ max_hamming bits. O(n²) on 64-bit ints (≈77 M pairs for 12.4 k images, vectorised).

    Returns columns: a, b (row indices), hamming, same_patient, split_a, split_b.
    """
    if hashes is None:
        it = df["path"]
        if progress:
            try:
                from tqdm.auto import tqdm

                it = tqdm(it, desc="dHash")
            except ImportError:
                pass
        hashes = np.array([dhash(p) for p in it], dtype=np.uint64)
    h = hashes.astype(np.uint64)
    rows = []
    # popcount via byte lookup
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

    def popcount(x: np.ndarray) -> np.ndarray:
        c = np.zeros(x.shape, dtype=np.uint8)
        for k in range(8):
            c += lut[((x >> np.uint64(8 * k)) & np.uint64(255)).astype(np.uint8)]
        return c

    for i in range(len(h) - 1):
        d = popcount(np.bitwise_xor(h[i], h[i + 1:]))
        js = np.nonzero(d <= max_hamming)[0]
        for j in js:
            rows.append((i, i + 1 + int(j), int(d[j])))
    out = pd.DataFrame(rows, columns=["a", "b", "hamming"])
    if len(out):
        out["same_patient"] = df["Patient_num"].to_numpy()[out["a"]] == df["Patient_num"].to_numpy()[out["b"]]
        if "split" in df:
            out["split_a"] = df["split"].to_numpy()[out["a"]]
            out["split_b"] = df["split"].to_numpy()[out["b"]]
    return out


# ----------------------------------------------------------------------------------------------
# statistics for the website (schema: DatasetStats)
# ----------------------------------------------------------------------------------------------

def dataset_stats(df: pd.DataFrame, duplicates: Optional[pd.DataFrame] = None, max_hamming: Optional[int] = None,
                  script: str = "omari_us.datasets.fetal_planes.dataset_stats") -> Dict:
    from ..utils import git_commit

    tr = df[df["Train"] == 1]
    te = df[df["Train"] == 0]
    counts = lambda d: [int((d["label"] == c).sum()) for c in range(NUM_CLASSES)]  # noqa: E731
    stats: Dict = {
        "provenance": {"script": script, "generated_at": now_iso(), "git_commit": git_commit(), "note": "Computed from FETAL_PLANES_DB_data.csv"},
        "total_images": int(len(df)),
        "total_patients": int(df["Patient_num"].nunique()),
        "classes": CLASSES,
        "class_counts": {"train": counts(tr), "test": counts(te)},
        "images_per_patient": df.groupby("Patient_num").size().astype(int).tolist(),
        "split": {
            "train_images": int(len(tr)), "test_images": int(len(te)),
            "train_patients": int(tr["Patient_num"].nunique()), "test_patients": int(te["Patient_num"].nunique()),
            "patient_overlap": int(len(set(tr["Patient_num"]) & set(te["Patient_num"]))),
        },
    }
    if "Brain_plane" in df:
        bp = df.loc[df["label"] == 1, "Brain_plane"].value_counts()
        stats["brain_planes"] = {str(k): int(v) for k, v in bp.items()}
    if "US_Machine" in df:
        stats["machines"] = {str(m): counts(g) for m, g in df.groupby("US_Machine")}
    if "Operator" in df:
        stats["operators"] = {str(k): int(v) for k, v in df["Operator"].value_counts().items()}
    if duplicates is not None:
        cross = 0
        if len(duplicates) and "split_a" in duplicates:
            ta = duplicates["split_a"].isin(["train", "val"]) & (duplicates["split_b"] == "test")
            tb = duplicates["split_b"].isin(["train", "val"]) & (duplicates["split_a"] == "test")
            cross = int((ta | tb).sum())
        involved = set(duplicates["a"]).union(duplicates["b"]) if len(duplicates) else set()
        stats["near_duplicates"] = {
            "method": "dHash 8x8", "threshold": int(max_hamming or 0),
            "groups": int(_count_groups(duplicates)), "images_in_groups": int(len(involved)), "cross_split_pairs": cross,
        }
    return stats


def _count_groups(pairs: pd.DataFrame) -> int:
    if not len(pairs):
        return 0
    parent: Dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["a"], pairs["b"]):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return len({find(x) for x in parent})


def save_split_manifest(df: pd.DataFrame, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in ["Image_name", "Patient_num", "Plane", "label", "Brain_plane", "Operator", "US_Machine", "Train", "split", "path"] if c in df]
    df[cols].to_csv(path, index=False)


def load_split_manifest(path: str | os.PathLike, data_root: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if data_root is not None:  # re-root image paths (e.g. manifest made on another machine)
        img_dir = find_image_dir(data_root)
        df["path"] = [str(img_dir / (n + ".png" if not str(n).endswith(".png") else n)) for n in df["Image_name"]]
    check_patient_disjoint(df)
    return df


# ----------------------------------------------------------------------------------------------
# torch dataset
# ----------------------------------------------------------------------------------------------

class FetalPlanesDataset:
    """Minimal torch-style dataset over a metadata frame. Returns (tensor, label, index)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        from PIL import Image

        r = self.df.iloc[i]
        with Image.open(r["path"]) as im:
            im = im.convert("L")
            x = self.transform(im) if self.transform else im
        return x, int(r["label"]), i


def synthetic_metadata(n_patients: int = 60, images_per_patient: Tuple[int, int] = (2, 8), seed: int = 0,
                       image_dir: Optional[str] = None, size: Tuple[int, int] = (160, 120)) -> pd.DataFrame:
    """Synthetic stand-in for *tests and smoke runs only*: random grey images with FETAL_PLANES_DB-like metadata.

    Never used for any number shown on the website.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_patients + 1):
        for k in range(int(rng.integers(*images_per_patient))):
            c = int(rng.integers(0, NUM_CLASSES))
            name = f"Patient{p:05d}_Plane{c + 1}_{k + 1}_of_9"
            rows.append({"Image_name": name, "Patient_num": p, "Plane": CLASSES[c], "Brain_plane": "Not A Brain",
                         "Operator": f"Op {int(rng.integers(1, 4))}", "US_Machine": ["Voluson E6", "Aloka"][p % 2],
                         "Train": int(p <= n_patients // 2), "label": c})
            if image_dir:
                Path(image_dir).mkdir(parents=True, exist_ok=True)
                arr = (rng.random((size[1], size[0])) * 80 + c * 25).clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(Path(image_dir) / f"{name}.png")
    df = pd.DataFrame(rows)
    if image_dir:
        df["path"] = [str(Path(image_dir) / f"{n}.png") for n in df["Image_name"]]
    return df
