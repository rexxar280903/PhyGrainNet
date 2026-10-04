"""Index competition photos: sample id, camera and physical scale per image.

Real file names (verified on Kaggle, 2026-10-04):

    Training-All_Photos_updated/Training-All_Photos_updated/Motorola_Edge_H030_01.jpg
    Training-All_Photos_updated/.../Motorola_Edge_60_fusion_H374_01.jpg
    Test_All_Photos/Test_All_Photos/iPhone14_HPC_Münster_BS6_9,0-10m (1).JPG
    Test_All_Photos/Test_All_Photos/iPhone16_HPC_Airbus BS10-4bis7 (2).JPG

while sample_submission.csv uses `HPC_Muenster_BS6_9_0-10m`. Matching is therefore
done on a normalised key (umlauts transliterated, punctuation removed).

Camera keys come from ppm_updated.csv (columns: phone, camera, width, height, ppm).
The PPM refers to the native sensor resolution; if a file was downscaled we
rescale the PPM by (actual long side / native long side).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from phygrainnet.constants import (
    IMAGE_EXTENSIONS,
    PPM_FILE,
    SAMPLE_SUBMISSION_FILE,
    TEST_PHOTO_DIR,
    TRAIN_LABELS_FILE,
    TRAIN_PHOTO_DIR,
)

_TRANSLIT = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"}
_VIEW_SUFFIX = re.compile(r"(\s*\(\d+\)|[_\-\s]\d{1,3})$")

# Extra spellings seen or expected in file names, mapped to the ppm.csv `phone` value.
CAMERA_ALIASES = {
    "samsung": "Samsung A52",
    "samsunga52": "Samsung A52",
    "sma525f": "Samsung A52",
    "a52": "Samsung A52",
    "motorolaedge20": "Motorola Edge",
    "iphone14": "iPhone 14",
    "iphone16": "iPhone 16",
}


def norm_key(text: str) -> str:
    text = "".join(_TRANSLIT.get(ch, ch) for ch in str(text))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def strip_view_suffix(stem: str) -> tuple[str, int | None]:
    m = _VIEW_SUFFIX.search(stem)
    if not m:
        return stem, None
    digits = re.sub(r"\D", "", m.group(0))
    return stem[: m.start()], int(digits) if digits else None


@dataclass
class CameraTable:
    table: pd.DataFrame  # phone, camera, width, height, ppm
    keys: dict[str, str]  # normalised key -> phone

    @classmethod
    def from_csv(cls, path: Path) -> "CameraTable":
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"phone", "width", "height", "ppm"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns {missing}; found {list(df.columns)}")
        keys: dict[str, str] = {}
        for _, row in df.iterrows():
            keys[norm_key(row["phone"])] = row["phone"]
            if "camera" in df.columns and isinstance(row["camera"], str):
                keys[norm_key(row["camera"])] = row["phone"]
        for alias, phone in CAMERA_ALIASES.items():
            if phone in set(df["phone"]):
                keys.setdefault(alias, phone)
        return cls(df, keys)

    def match_prefix(self, stem_key: str) -> tuple[str | None, str]:
        """Longest camera key that prefixes the normalised stem."""
        best = None
        for key in self.keys:
            if stem_key.startswith(key) and (best is None or len(key) > len(best)):
                best = key
        if best is not None:
            return self.keys[best], stem_key[len(best):]
        # Fallback: camera name somewhere else in the file name (e.g. "H405_SamsungA52").
        inside = [k for k in self.keys if len(k) >= 4 and k in stem_key]
        if not inside:
            return None, stem_key
        k = max(inside, key=len)
        return self.keys[k], stem_key.replace(k, "", 1)

    def row(self, phone: str) -> pd.Series:
        return self.table.loc[self.table["phone"] == phone].iloc[0]


def match_sample_id(rest_key: str, known_ids: list[str], exact_only: bool = False) -> str | None:
    known = {norm_key(s): s for s in known_ids}
    if rest_key in known:
        return known[rest_key]
    if exact_only:
        return None
    hits = [k for k in known if k and k in rest_key]
    if not hits:
        return None
    return known[max(hits, key=len)]


def parse_photo_name(stem: str, cams: "CameraTable", known_ids: list[str]) -> tuple[str | None, str | None, int | None]:
    """Return (camera phone, sample_id, view index) for a file stem.

    The full stem is tried first so ids ending in digits (`Kleinkummerfeld 2-2`) are
    never truncated; then the stem without its view suffix (`_01`, ` (3)`).
    """
    stripped, view = strip_view_suffix(stem)
    for candidate, idx in ((stripped, view), (stem, None)):
        phone, rest = cams.match_prefix(norm_key(candidate))
        sid = match_sample_id(rest, known_ids, exact_only=True)
        if sid is not None:
            return phone, sid, idx
    phone, rest = cams.match_prefix(norm_key(stripped))
    return phone, match_sample_id(rest, known_ids), view


def find_dir(root: Path, name: str) -> Path | None:
    """Return the deepest directory called `name` (Kaggle nests folders twice)."""
    cands = [p for p in root.rglob(name) if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: len(p.parts))


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def build_catalog(
    data_root: str | Path,
    scale_mode: str = "resize",
    read_sizes: bool = True,
) -> pd.DataFrame:
    """One row per photo with split, sample_id, camera, view index and effective PPM.

    scale_mode:
        "resize" (default) - PPM rescaled by actual/native long side (correct if files
                             were downscaled; identical to native when not).
        "native"           - always use the table PPM (correct if files were cropped).
    """
    root = Path(data_root)
    cams = CameraTable.from_csv(root / PPM_FILE)
    train_ids = pd.read_csv(root / TRAIN_LABELS_FILE)["sample_id"].astype(str).tolist()
    test_ids = pd.read_csv(root / SAMPLE_SUBMISSION_FILE)["sample_id"].astype(str).tolist()

    rows = []
    for split, folder_name, ids in (("train", TRAIN_PHOTO_DIR, train_ids), ("test", TEST_PHOTO_DIR, test_ids)):
        folder = find_dir(root, folder_name)
        if folder is None:
            raise FileNotFoundError(f"photo folder {folder_name!r} not found under {root}")
        for path in list_images(folder):
            phone, sample_id, view = parse_photo_name(path.stem, cams, ids)
            rec = {
                "path": str(path),
                "file": path.name,
                "split": split,
                "sample_id": sample_id,
                "camera": phone,
                "view": view,
            }
            if phone is not None:
                cam = cams.row(phone)
                rec.update(native_w=int(cam["width"]), native_h=int(cam["height"]), ppm_native=float(cam["ppm"]))
            if read_sizes:
                with Image.open(path) as im:
                    rec.update(width=im.width, height=im.height)
            rows.append(rec)

    df = pd.DataFrame(rows)
    if read_sizes and "ppm_native" in df:
        native_long = df[["native_w", "native_h"]].max(axis=1)
        actual_long = df[["width", "height"]].max(axis=1)
        df["resize_factor"] = actual_long / native_long
        if scale_mode == "resize":
            df["ppm"] = df["ppm_native"] * df["resize_factor"]
        elif scale_mode == "native":
            df["ppm"] = df["ppm_native"]
        else:
            raise ValueError(f"unknown scale_mode {scale_mode!r}")
        df["field_w_mm"] = df["width"] / df["ppm"]
        df["field_h_mm"] = df["height"] / df["ppm"]
    return df


def catalog_problems(df: pd.DataFrame, train_ids: list[str], test_ids: list[str]) -> list[str]:
    problems = []
    for col in ("sample_id", "camera"):
        bad = df[df[col].isna()]
        if len(bad):
            problems.append(f"{len(bad)} files without {col}: {bad['file'].tolist()[:10]}")
    for split, ids in (("train", train_ids), ("test", test_ids)):
        seen = set(df.loc[df["split"] == split, "sample_id"].dropna())
        missing = sorted(set(ids) - seen)
        if missing:
            problems.append(f"{split}: no photos for {missing}")
    if "resize_factor" in df:
        odd = df[(df["resize_factor"] - 1).abs() > 0.02]
        if len(odd):
            problems.append(
                f"{len(odd)} files differ from native size (resize factors "
                f"{sorted(odd['resize_factor'].round(3).unique().tolist())[:8]}); check scale_mode"
            )
    return problems
