from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

from Utils.corpus import load_corpus_item

def _apply_mask_remove(img: Image.Image, mask_bool: np.ndarray) -> Image.Image:
    """
    Remove without inpainting: set masked pixels to black.
    """
    arr = np.array(img).copy()
    arr[mask_bool] = [0, 0, 0]
    return Image.fromarray(arr)

def _paste_rgba(base_rgb: Image.Image, patch_rgba: Image.Image, x: int, y: int) -> Image.Image:
    out = base_rgb.copy()
    if patch_rgba.mode != "RGBA":
        patch_rgba = patch_rgba.convert("RGBA")
    out.paste(patch_rgba, (x, y), patch_rgba.split()[3])
    return out

def _fit_patch_to_box(patch_rgba: Image.Image, bbox, scale: float) -> Image.Image:
    x1,y1,x2,y2 = bbox
    bw = max(1, x2-x1)
    bh = max(1, y2-y1)
    # scale relative to bbox
    tw = max(1, int(bw * scale))
    th = max(1, int(bh * scale))
    return patch_rgba.resize((tw, th), Image.BILINEAR)

def apply_sa(
    img_sp: Image.Image,
    sa_type: int,
    yolo_dets,              # list of Det
    chosen_idx: int,
    sam_segmenter,
    object_corpus_dir: str,
    ins_corpus_id: int,
    ins_scale: float,
    rep_corpus_id: int,
    rep_scale: float,
) -> (Image.Image, Dict[str, Any]):
    """
    Returns (final_image, sa_log_dict)
    sa_type: 1 insert, 2 remove, 3 replace
    """
    log: Dict[str, Any] = {"sa_type": int(sa_type)}

    if sa_type == 1:
        # INSERT: choose corpus object and paste somewhere simple (e.g., top-left of chosen bbox)
        item = load_corpus_item(object_corpus_dir, ins_corpus_id)
        # if no dets, just paste in corner
        if len(yolo_dets) > 0:
            det = yolo_dets[min(chosen_idx, len(yolo_dets)-1)]
            x1,y1,x2,y2 = det.bbox_xyxy
            patch = _fit_patch_to_box(item.obj_img, det.bbox_xyxy, ins_scale)
            img_out = _paste_rgba(img_sp, patch, x1, y1)
            log.update({"inserted": {"corpus_id": item.corpus_id, "class_name": item.class_name, "at_bbox": det.bbox_xyxy}})
        else:
            patch = item.obj_img.resize((64,64))
            img_out = _paste_rgba(img_sp, patch, 0, 0)
            log.update({"inserted": {"corpus_id": item.corpus_id, "class_name": item.class_name, "at_bbox": None}})
        return img_out, log

    # REMOVE / REPLACE require a detection
    det = yolo_dets[chosen_idx]
    bbox = det.bbox_xyxy
    mask = sam_segmenter.mask_from_box(img_sp, bbox)

    log["target_det"] = {"idx": int(chosen_idx), "cls_id": det.cls_id, "cls_name": det.cls_name, "conf": det.conf, "bbox_xyxy": bbox}

    img_removed = _apply_mask_remove(img_sp, mask)

    if sa_type == 2:
        return img_removed, log

    if sa_type == 3:
        item = load_corpus_item(object_corpus_dir, rep_corpus_id)
        patch = _fit_patch_to_box(item.obj_img, bbox, rep_scale)
        x1,y1,_,_ = bbox
        img_out = _paste_rgba(img_removed, patch, x1, y1)
        log["replaced_with"] = {"corpus_id": item.corpus_id, "class_name": item.class_name}
        return img_out, log

    # sa_type == 0 or unknown
    return img_sp, log