from pathlib import Path
from PIL import Image
import json

def _get_object_dirs(corpus_dir: str):
    root = Path(corpus_dir)

    # support both:
    # object_corpus/objects/00000000/
    # or directly object_corpus/00000000/
    objects_root = root / "objects"
    base = objects_root if objects_root.exists() else root

    dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    return dirs

def corpus_size(corpus_dir: str) -> int:
    return len(_get_object_dirs(corpus_dir))

def load_corpus_item(corpus_dir: str, idx: int) -> Image.Image:
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]
    img_path = obj_dir / "obj.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Missing obj.png in {obj_dir}")

    return Image.open(img_path).convert("RGB")

def load_corpus_mask(corpus_dir: str, idx: int):
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]
    mask_path = obj_dir / "mask.png"
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask.png in {obj_dir}")

    return Image.open(mask_path).convert("L")

def load_corpus_meta(corpus_dir: str, idx: int):
    dirs = _get_object_dirs(corpus_dir)
    if not (0 <= idx < len(dirs)):
        raise IndexError(f"Corpus idx {idx} out of range, size={len(dirs)}")

    obj_dir = dirs[idx]
    meta_path = obj_dir / "meta.json"
    if not meta_path.exists():
        return {}

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)