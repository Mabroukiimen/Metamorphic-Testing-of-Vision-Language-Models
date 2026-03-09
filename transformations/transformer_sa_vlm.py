from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from Utils.corpus import load_corpus_item  # you may need to create this helper


def pil_to_bgr(img: Image.Image):
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def bgr_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def clip_idx(idx: int, n: int) -> int:
    if n <= 0:
        raise ValueError("Empty list")
    return min(max(0, idx), n - 1)


def segment_from_box(sam_segmenter, img_pil: Image.Image, bbox_xyxy):
    return sam_segmenter.mask_from_box(img_pil, bbox_xyxy)  
    # should return binary mask HxW


def extract_object_from_mask(img_bgr, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Empty mask")
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    crop = img_bgr[y1:y2+1, x1:x2+1].copy()
    crop_mask = mask[y1:y2+1, x1:x2+1].copy()
    crop = crop * crop_mask[..., None]
    return crop, crop_mask, [int(x1), int(y1), int(x2), int(y2)]


def paste_object(background, obj_crop, obj_mask, x, y):
    bg = background.copy()
    h, w = obj_crop.shape[:2]
    H, W = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)
    if x1 >= x2 or y1 >= y2:
        return bg

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    roi = bg[y1:y2, x1:x2]
    obj = obj_crop[oy1:oy2, ox1:ox2]
    mask = obj_mask[oy1:oy2, ox1:ox2].astype(np.float32)[..., None]

    bg[y1:y2, x1:x2] = (roi * (1 - mask) + obj * mask).astype(np.uint8)
    return bg


def choose_insertion_xy(img_bgr, yolo_dets):
    H, W = img_bgr.shape[:2]
    if len(yolo_dets) == 0:
        return W // 3, H // 3

    det = yolo_dets[0]
    x1, y1, x2, y2 = det.bbox_xyxy
    return int(x2 + 10), int(y1)


def apply_insert(
    img_sp: Image.Image,
    yolo_dets,
    sam_segmenter,
    object_corpus_dir: str,
    ins_corpus_id: int,
    ins_scale: float,
):
    corpus_img_pil = load_corpus_item(object_corpus_dir, ins_corpus_id)
    corpus_bgr = pil_to_bgr(corpus_img_pil)

    corpus_dets = sam_segmenter.yolo.detect_topk(corpus_img_pil, topk=1)  # or another helper
    if len(corpus_dets) == 0:
        raise ValueError("No object detected in corpus item")

    src_det = corpus_dets[0]
    src_mask = segment_from_box(sam_segmenter, corpus_img_pil, src_det.bbox_xyxy)
    obj_crop, obj_mask, src_bbox = extract_object_from_mask(corpus_bgr, src_mask)

    h, w = obj_crop.shape[:2]
    new_w = max(1, int(w * ins_scale))
    new_h = max(1, int(h * ins_scale))

    obj_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    bg_bgr = pil_to_bgr(img_sp)
    ins_x, ins_y = choose_insertion_xy(bg_bgr, yolo_dets)
    out_bgr = paste_object(bg_bgr, obj_crop, obj_mask, ins_x, ins_y)

    log = {
        "sa_type": 1,
        "inserted_corpus_id": ins_corpus_id,
        "insert_scale": float(ins_scale),
        "insert_position": [int(ins_x), int(ins_y)],
        "source_bbox": src_bbox,
    }
    return bgr_to_pil(out_bgr), log


def apply_remove(
    img_sp: Image.Image,
    yolo_dets,
    chosen_idx: int,
    sam_segmenter,
    lama_inpainter,
):
    if len(yolo_dets) == 0:
        raise ValueError("No detections for removal")

    chosen_idx = clip_idx(chosen_idx, len(yolo_dets))
    det = yolo_dets[chosen_idx]
    mask = segment_from_box(sam_segmenter, img_sp, det.bbox_xyxy)

    img_bgr = pil_to_bgr(img_sp)
    out_bgr = lama_inpainter.inpaint(img_bgr, mask)  # your LaMa wrapper

    log = {
        "sa_type": 2,
        "target_det_idx": int(chosen_idx),
        "removed_bbox": list(map(int, det.bbox_xyxy)),
        "removed_cls_name": det.cls_name,
        "mask_area": int(mask.sum()),
    }
    return bgr_to_pil(out_bgr), log


def apply_sa(
    img_sp: Image.Image,
    sa_type: int,
    yolo_dets,
    chosen_idx: int,
    sam_segmenter,
    lama_inpainter,
    object_corpus_dir: str,
    ins_corpus_id=None,
    ins_scale=None,
    rep_corpus_id=None,
    rep_scale=None,
):
    if sa_type == 0:
        return img_sp, {"sa_type": 0}

    if sa_type == 1:
        return apply_insert(
            img_sp=img_sp,
            yolo_dets=yolo_dets,
            sam_segmenter=sam_segmenter,
            object_corpus_dir=object_corpus_dir,
            ins_corpus_id=ins_corpus_id,
            ins_scale=ins_scale,
        )

    if sa_type == 2:
        return apply_remove(
            img_sp=img_sp,
            yolo_dets=yolo_dets,
            chosen_idx=chosen_idx,
            sam_segmenter=sam_segmenter,
            lama_inpainter=lama_inpainter,
        )

    raise NotImplementedError("Replace not added yet")