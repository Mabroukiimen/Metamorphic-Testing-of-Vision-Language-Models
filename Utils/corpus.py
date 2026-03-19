from pathlib import Path
from PIL import Image
import json


def _get_object_dirs(corpus_dir: str):
    root = Path(corpus_dir)
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs


def _get_single_png(obj_dir: Path) -> Path:
    pngs = sorted(obj_dir.glob("*.png"))
    if len(pngs) == 0:
        raise FileNotFoundError(f"No PNG found in {obj_dir}")
    return pngs[0]


def corpus_size(corpus_dir: str) -> int:
    return len(_get_object_dirs(corpus_dir))


def load_corpus_item(corpus_dir: str, idx: int) -> Image.Image:
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]
    png_path = _get_single_png(obj_dir)

    rgba = Image.open(png_path).convert("RGBA")
    alpha = rgba.getchannel("A")
    
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError(f"Empty alpha mask in {png_path}")
    
    rgba = rgba.crop(bbox)
    alpha = alpha.crop(bbox)
    
    rgb = rgba.convert("RGB")
    return rgb


def load_corpus_mask(corpus_dir: str, idx: int):
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]
    png_path = _get_single_png(obj_dir)

    rgba = Image.open(png_path).convert("RGBA")
    alpha = rgba.getchannel("A")
    
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError(f"Empty alpha mask in {png_path}")
    
    alpha = alpha.crop(bbox)
    
    return alpha.convert("L")


def load_corpus_meta(corpus_dir: str, idx: int):
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]

    meta_path = obj_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return {"class_name": obj_dir.name}