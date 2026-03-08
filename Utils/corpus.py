from dataclasses import dataclass
from pathlib import Path
import json
from PIL import Image

@dataclass
class CorpusItem:
    corpus_id: int
    class_id: int
    class_name: str
    obj_img: Image.Image  # RGBA preferred
    meta: dict

def _id_to_dir(root: Path, corpus_id: int) -> Path:
    # supports either "000123" folders or "123"
    p1 = root / f"{corpus_id:06d}"
    if p1.exists():
        return p1
    p2 = root / str(corpus_id)
    return p2

def load_corpus_item(corpus_dir: str, corpus_id: int) -> CorpusItem:
    root = Path(corpus_dir)
    d = _id_to_dir(root, corpus_id)
    meta_path = d / "meta.json"
    obj_path = d / "obj.png"

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    img = Image.open(obj_path)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    return CorpusItem(
        corpus_id=int(meta.get("corpus_id", corpus_id)),
        class_id=int(meta.get("class_id", -1)),
        class_name=str(meta.get("class_name", "UNKNOWN")),
        obj_img=img,
        meta=meta,
    )

def corpus_size(corpus_dir: str) -> int:
    root = Path(corpus_dir)
    # count folders that contain meta.json
    return sum(1 for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())