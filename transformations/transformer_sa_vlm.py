from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from Utils.corpus import load_corpus_item, load_corpus_mask, load_corpus_meta
from Utils.psnr import compute_psnr_pil
from transformations.transformer_sp_vlm import SPTransformer


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


def choose_insertion_xy(img_bgr, yolo_dets, obj_w, obj_h, max_iou=0.15):
    H, W = img_bgr.shape[:2]

    candidate_positions = []

    if len(yolo_dets) > 0:
        for det in yolo_dets:
            x1, y1, x2, y2 = det.bbox_xyxy

            candidate_positions.extend([
                (int(x2 + 10), int(y1)),              # right
                (int(x1 - obj_w - 10), int(y1)),      # left
                (int(x1), int(y2 + 10)),              # below
                (int(x1), int(y1 - obj_h - 10)),      # above
            ])

    candidate_positions.extend([
        (W // 3, H // 3),
        (W // 2, H // 2),
        (max(0, W - obj_w - 20), max(0, H - obj_h - 20)),
    ])

    for x, y in candidate_positions:
        x = max(0, min(x, W - obj_w))
        y = max(0, min(y, H - obj_h))

        insert_box = [x, y, x + obj_w, y + obj_h]

        if not too_much_overlap(insert_box, yolo_dets, max_iou=max_iou):
            return x, y

    # fallback if all candidates overlap too much
    x = max(0, min(W // 3, W - obj_w))
    y = max(0, min(H // 3, H - obj_h))
    return x, y


def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def too_much_overlap(insert_box, yolo_dets, max_iou=0.15):
    for det in yolo_dets:
        det_box = det.bbox_xyxy
        if box_iou_xyxy(insert_box, det_box) > max_iou:
            return True
    return False


def apply_insert(
    img_sp: Image.Image,
    yolo_dets,
    sam_segmenter,
    object_corpus_dir: str,
    ins_corpus_id: int,
    ins_scale: float,
):
    # load pre-cut object and its mask directly from the corpus
    corpus_img_pil = load_corpus_item(object_corpus_dir, ins_corpus_id)
    corpus_mask_pil = load_corpus_mask(object_corpus_dir, ins_corpus_id)
    corpus_meta = load_corpus_meta(object_corpus_dir, ins_corpus_id)

    obj_crop = pil_to_bgr(corpus_img_pil)
    obj_mask = (np.array(corpus_mask_pil) > 0).astype(np.uint8)
    inserted_class_name = corpus_meta.get("class_name", "unknown")

    if ins_scale is None or ins_scale <= 0:
        raise ValueError(f"Invalid ins_scale: {ins_scale}")

    h, w = obj_crop.shape[:2]
    new_w = max(1, int(w * ins_scale))
    new_h = max(1, int(h * ins_scale))

    obj_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    bg_bgr = pil_to_bgr(img_sp)
    
    ins_x, ins_y = choose_insertion_xy(
        bg_bgr,
        yolo_dets,
        obj_w=new_w,
        obj_h=new_h,
        max_iou=0.15,
    )
    out_bgr = paste_object(bg_bgr, obj_crop, obj_mask, ins_x, ins_y)

    log = {
        "sa_type": 1,
        "inserted_corpus_id": int(ins_corpus_id),
        "inserted_class_name": inserted_class_name,
        "insert_scale": float(ins_scale),
        "insert_position": [int(ins_x), int(ins_y)],
        "inserted_mask_area": int(obj_mask.sum()),
    
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

def apply_replace(
    img_sp: Image.Image,
    yolo_dets,
    chosen_idx: int,
    sam_segmenter,
    lama_inpainter,
    object_corpus_dir: str,
    rep_corpus_id: int,
    rep_scale: float,
):
    if len(yolo_dets) == 0:
        raise ValueError("No detections for replacement")

    chosen_idx = clip_idx(chosen_idx, len(yolo_dets))
    det = yolo_dets[chosen_idx]

    # target object in original image
    target_mask = segment_from_box(sam_segmenter, img_sp, det.bbox_xyxy)
    img_bgr = pil_to_bgr(img_sp)

    # remove target object first
    removed_bgr = lama_inpainter.inpaint(img_bgr, target_mask)

    # load replacement object from corpus
    corpus_img_pil = load_corpus_item(object_corpus_dir, rep_corpus_id)
    corpus_mask_pil = load_corpus_mask(object_corpus_dir, rep_corpus_id)
    corpus_meta = load_corpus_meta(object_corpus_dir, rep_corpus_id)

    obj_crop = pil_to_bgr(corpus_img_pil)
    obj_mask = (np.array(corpus_mask_pil) > 0).astype(np.uint8)
    replacement_class_name = corpus_meta.get("class_name", "unknown")

    if rep_scale is None or rep_scale <= 0:
        raise ValueError(f"Invalid rep_scale: {rep_scale}")

    # scale relative to target bbox
    x1, y1, x2, y2 = det.bbox_xyxy
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)

    base_size = max(target_w, target_h)
    h, w = obj_crop.shape[:2]

    if h == 0 or w == 0:
        raise ValueError("Replacement object has invalid size")

    long_side = max(h, w)
    scale_factor = (base_size * rep_scale) / long_side

    new_w = max(1, int(w * scale_factor))
    new_h = max(1, int(h * scale_factor))

    obj_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    obj_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # place replacement near center of removed target bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    paste_x = int(cx - new_w // 2)
    paste_y = int(cy - new_h // 2)

    out_bgr = paste_object(removed_bgr, obj_crop, obj_mask, paste_x, paste_y)

    log = {
        "sa_type": 3,
        "target_det_idx": int(chosen_idx),
        "replaced_bbox": list(map(int, det.bbox_xyxy)),
        "replaced_cls_name": det.cls_name,
        "replacement_corpus_id": int(rep_corpus_id),
        "replacement_class_name": replacement_class_name,
        "replacement_scale": float(rep_scale),
        "replacement_position": [int(paste_x), int(paste_y)],
        "replacement_mask_area": int(obj_mask.sum()),
    }
    return bgr_to_pil(out_bgr), log


def apply_scale_object(
    img_sp: Image.Image,
    yolo_dets,
    chosen_idx: int,
    sam_segmenter,
    lama_inpainter,
    obj_scale_factor: float,
):
    
    print("ENTERED apply_scale_object")
    print("lama_inpainter is None?", lama_inpainter is None)
    print("obj_scale_factor =", obj_scale_factor)
    
    if len(yolo_dets) == 0:
        raise ValueError("No detections for scale_object")

    if obj_scale_factor is None or obj_scale_factor <= 0:
        raise ValueError(f"Invalid obj_scale_factor: {obj_scale_factor}")

    chosen_idx = clip_idx(chosen_idx, len(yolo_dets))
    det = yolo_dets[chosen_idx]

    target_mask = segment_from_box(sam_segmenter, img_sp, det.bbox_xyxy)
    img_bgr = pil_to_bgr(img_sp)

    # remove original object first
    removed_bgr = lama_inpainter.inpaint(img_bgr, target_mask)

    # extract original object crop from the image using the mask
    obj_crop, obj_mask, obj_bbox = extract_object_from_mask(img_bgr, target_mask)

    h, w = obj_crop.shape[:2]
    new_w = max(1, int(round(w * obj_scale_factor)))
    new_h = max(1, int(round(h * obj_scale_factor)))

    scaled_crop = cv2.resize(obj_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    scaled_mask = cv2.resize(obj_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    x1, y1, x2, y2 = det.bbox_xyxy
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    paste_x = int(cx - new_w // 2)
    paste_y = int(cy - new_h // 2)

    out_bgr = paste_object(removed_bgr, scaled_crop, scaled_mask, paste_x, paste_y)

    log = {
        "sa_type": 5,
        "sa_type_name": "scale_object",
        "target_det_idx": int(chosen_idx),
        "target_bbox": list(map(int, det.bbox_xyxy)),
        "target_cls_name": det.cls_name,
        "obj_scale_factor": float(obj_scale_factor),
        "scaled_position": [int(paste_x), int(paste_y)],
        "scaled_mask_area": int(scaled_mask.sum()),
    }

    return bgr_to_pil(out_bgr), log

def apply_object_local_sp(   #no global SP first #no global SP first #PSNR is computed on the object crop only
    img_sp: Image.Image,
    tr_vector,
    yolo_dets,
    chosen_idx: int,
    sam_segmenter,
    psnr_min: float = 20.0,
):
    if len(yolo_dets) == 0:
        raise ValueError("No detections for object_local_sp")

    chosen_idx = clip_idx(chosen_idx, len(yolo_dets))
    det = yolo_dets[chosen_idx]

    mask = segment_from_box(sam_segmenter, img_sp, det.bbox_xyxy)
    img_bgr = pil_to_bgr(img_sp)

    obj_crop_bgr, obj_mask, obj_bbox = extract_object_from_mask(img_bgr, mask)
    obj_crop_rgb = cv2.cvtColor(obj_crop_bgr, cv2.COLOR_BGR2RGB)
    obj_crop_pil = Image.fromarray(obj_crop_rgb)

    # apply SP only on the object crop
    tr_px = SPTransformer(obj_crop_pil.copy())
    obj_after_pixel = tr_px.apply_pixel(tr_vector)

    obj_before_arr = np.array(obj_crop_pil.convert("RGB"))
    obj_after_arr = np.array(obj_after_pixel.convert("RGB"))
    
    obj_before_arr = obj_before_arr * obj_mask[..., None]
    obj_after_arr = obj_after_arr * obj_mask[..., None]
    
    obj_before_masked = Image.fromarray(obj_before_arr.astype(np.uint8))
    obj_after_masked = Image.fromarray(obj_after_arr.astype(np.uint8))
    
    object_psnr = compute_psnr_pil(obj_before_masked, obj_after_masked)
    
    if object_psnr < psnr_min:
        return None, {
            "sa_type": 4,
            "sa_type_name": "object_local_sp",
            "target_det_idx": int(chosen_idx),
            "target_bbox": list(map(int, det.bbox_xyxy)),
            "target_cls_name": det.cls_name,
            "object_psnr": float(object_psnr),
            "rejected": True,
            "reject_reason": "object_psnr_below_threshold",
        }

    tr_geo = SPTransformer(obj_after_pixel.copy())
    obj_final_pil = tr_geo.apply_geometric(tr_vector)

    obj_final_bgr = cv2.cvtColor(np.array(obj_final_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    obj_final_bgr = obj_final_bgr * obj_mask[..., None]

    x1, y1, _, _ = obj_bbox
    out_bgr = paste_object(img_bgr, obj_final_bgr, obj_mask, x1, y1)

    log = {
        "sa_type": 4,
        "sa_type_name": "object_local_sp",
        "target_det_idx": int(chosen_idx),
        "target_bbox": list(map(int, det.bbox_xyxy)),
        "target_cls_name": det.cls_name,
        "object_bbox": list(map(int, obj_bbox)),
        "object_psnr": float(object_psnr),
        "rejected": False,
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
    tr_vector=None,
    ins_corpus_id=None,
    ins_scale=None,
    rep_corpus_id=None,
    rep_scale=None,
    obj_scale_factor=None,
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

    if sa_type == 3:
        return apply_replace(
            img_sp=img_sp,
            yolo_dets=yolo_dets,
            chosen_idx=chosen_idx,
            sam_segmenter=sam_segmenter,
            lama_inpainter=lama_inpainter,
            object_corpus_dir=object_corpus_dir,
            rep_corpus_id=rep_corpus_id,
            rep_scale=rep_scale,
        )
        
    if sa_type == 4:
        return apply_object_local_sp(
            img_sp=img_sp,
            tr_vector=tr_vector,
            yolo_dets=yolo_dets,
            chosen_idx=chosen_idx,
            sam_segmenter=sam_segmenter,
            psnr_min=20.0,
        )
        
    if sa_type == 5:
        return apply_scale_object(
            img_sp=img_sp,
            yolo_dets=yolo_dets,
            chosen_idx=chosen_idx,
            sam_segmenter=sam_segmenter,
            lama_inpainter=lama_inpainter,
            obj_scale_factor=obj_scale_factor,
        )

    raise ValueError(f"Unsupported sa_type: {sa_type}")