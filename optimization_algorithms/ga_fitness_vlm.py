import json
from pathlib import Path
from typing import Sequence, Dict, Any
from PIL import Image

from Utils.vector_layout import Vec
from Utils.vector_decoder_vlm import VectorDecoderVLM
from Utils.psnr import compute_psnr_pil
from Utils.corpus import corpus_size

from transformations.transformer_sa_vlm import apply_sa
from judge.llm_judge import judge_score_row

import os
print("PID:", os.getpid())


#dets: detections from sp_img
#base_yolo_dets: detections from base_img
#we match the bbox of the chosen det in sp_img to the base_img det with highest IoU, we then use the matched base det class name as the "target class" that the SA is supposed to remove or replace, and we check the transformed caption to see if it successfully removed/replaced that target class.

#chosen_idx is the index of the selected detection in dets, which is determined by the gene TARGET_DET_IDX in the GA vector. The system will try to remove/replace the object corresponding to that detection, so it’s like the “target object” for the SA. We want to make sure that the target object is actually present in the image, so we use YOLO detections to find it. If there are no detections, then we can’t really perform a removal or replacement, so we return a bad fitness score. If there are detections, we clip the chosen_idx to be within the range of available detections. Then we take the bounding box of that chosen detection in the sp_img, and we find which detection in the base_img has the highest IoU with that bounding box. We consider that matched detection in the base_img as the “corresponding object” that we are trying to remove or replace. We use its class name as the target class for judging whether the SA successfully removed or replaced it in the caption.

def bbox_iou_xyxy(boxA, boxB) -> float:    #after choosing chosen_idx on sp_img, match its bbox to the base-image bbox with highest IoU
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = areaA + areaB - inter_area
    if union <= 0:
        return 0.0

    return inter_area / union


def match_sp_det_to_base_det(sp_det, base_dets):
    if not base_dets:
        return None, -1, 0.0

    best_idx = -1
    best_iou = -1.0

    for i, bdet in enumerate(base_dets):
        iou = bbox_iou_xyxy(sp_det.bbox_xyxy, bdet.bbox_xyxy)
        if iou > best_iou:
            best_iou = iou
            best_idx = i

    if best_idx == -1:
        return None, -1, 0.0

    return base_dets[best_idx], best_idx, best_iou

def vlm_mt_fitness(tr_vector: Sequence[float], *args, **kwargs) -> float:
    
    cfg = kwargs["cfg"]
    paths = cfg["paths"]
    thr = cfg["thresholds"]
    llm_cfg = cfg.get("llm", {})

    base_image_path: str = kwargs["base_image_path"]
    image_id = kwargs.get("image_id")

    vlm_runner = kwargs["vlm_runner"]
    sts = kwargs["sts_scorer"]
    
    yolo = kwargs["yolo_detector"]
    sam  = kwargs["sam_segmenter"]
    lama = kwargs.get("lama_inpainter", None)

    # Dict[str, Any] = kwargs.setdefault("cache", {})
    cache: Dict[str, Any] = kwargs["cache"]
    print("CACHE_ID:", id(cache))
    print("fitness call, cache keys:", list(cache.keys())[:5])
    log_path = Path(paths["ga_log_path"])
    out_dir = Path(paths["transformed_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    # cache base image
    if "base_img" not in cache:
        cache["base_img"] = Image.open(base_image_path).convert("RGB")
    base_img = cache["base_img"]
    
    topk = int(thr.get("yolo_topk", 10))
    
    if "base_yolo_dets" not in cache:
        cache["base_yolo_dets"] = yolo.detect_topk(base_img, topk=topk)
        
    base_yolo_dets = cache["base_yolo_dets"]
    base_object_classes = sorted(list({d.cls_name for d in base_yolo_dets}))
    

    # cache base caption
    if "base_caption" not in cache:
        cache["base_caption"] = vlm_runner.caption(base_image_path, prompt=cfg["vlm"]["caption_prompt"])
        print("BASE CAPTION USED:", cache["base_caption"])
    base_caption = cache["base_caption"]

    v = list(tr_vector)
    
    decoder = VectorDecoderVLM(base_img)
    
    px_img = decoder.apply_pixel(v)
    psnr_val = compute_psnr_pil(base_img, px_img)
    psnr_min = float(thr.get("psnr_min", 20.0))
    if psnr_val < psnr_min:
        '''
        GA is set up to minimize the fitness value.
        So if a candidate is “invalid” (PSNR < threshold), we want it to look very bad to the GA.
        Returning a huge value like 1e6 makes that individual have terrible fitness, so it won’t be selected and won’t survive.'''
        return 1e6  
    
    sp_img = decoder.apply_geometric(px_img, v)

    # SA selector
    sa_type = int(round(v[Vec.SA_TYPE]))
    sa_type = max(0, min(3, sa_type))
    
    #sa_type = 0  # TEMP: run SP-only (no YOLO/SAM/LLM judge)

    # prepare logging shell
    eval_idx = cache.get("eval_idx", 0)
    cache["eval_idx"] = eval_idx + 1
    
    
    
    #logs
    record: Dict[str, Any] = {
        "image": base_image_path,
        "image_id": image_id,
        "vector": list(map(float, v)),
        "base_caption": base_caption,
        "base_image_object_classes": base_object_classes,
        "psnr_pixel": psnr_val,
        "sp": {
            "b_bright": int(round(v[Vec.B_BRIGHT])),
            "bright_factor": float(v[Vec.BRIGHT_FACTOR]),
            "b_blur": int(round(v[Vec.B_BLUR])),
            "blur_radius": float(v[Vec.BLUR_RADIUS]),
            "b_rotate": int(round(v[Vec.B_ROTATE])),
            "rot_angle": float(v[Vec.ROT_ANGLE]),
            "b_translate": int(round(v[Vec.B_TRANSLATE])),
            "tx": int(v[Vec.TX]),
            "ty": int(v[Vec.TY]),
        },
        "sa": {"sa_type": sa_type},
        "eval_idx": eval_idx,
    }

    # If SA is none: STS fitness
    if sa_type == 0:
        tmp_path = out_dir / f"img{image_id}_eval{eval_idx:04d}_sp.png"
        sp_img.save(tmp_path)
        print(f"\n[EVAL {eval_idx:04d}] saved: {tmp_path.resolve()}")
        print(f"[EVAL {eval_idx:04d}] vector: {list(map(float, v))}")
        trf_caption = vlm_runner.caption(str(tmp_path.resolve()), prompt=cfg["vlm"]["caption_prompt"])
        print("transformed caption:", trf_caption)
        dist = sts.distance(base_caption, trf_caption)
        record["transformed_image_path"] = str(tmp_path.resolve())
        record["transformed_caption"] = trf_caption
        record["sts_distance"] = float(dist)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # minimize (-dist) -> maximize dist
        return -float(dist)

    # ---- SA path ----
    dets = yolo.detect_topk(sp_img, topk=topk)
    record["sa"]["yolo_topk"] = [
        {"cls_id": d.cls_id, "cls_name": d.cls_name, "conf": d.conf, "bbox_xyxy": list(d.bbox_xyxy)}
        for d in dets
    ]

    # If need target but no dets -> reject
    if (sa_type in (2, 3)) and len(dets) == 0:
        return 1e6

    # clip target_det_idx
    target_gene = int(round(v[Vec.TARGET_DET_IDX]))    #TARGET_DET_IDX is the gene in the GA vector that tells the system which detected object to choose from YOLO’s detections, in case of removal or replacement.
    if len(dets) > 0:
        chosen_idx = min(max(0, target_gene), len(dets) - 1)
    else:
        chosen_idx = 0  # for insertion, can ignore target
        
    matched_base_det = None
    matched_base_idx = -1
    matched_base_iou = 0.0
    matched_base_cls_name = "UNKNOWN"
    
    if sa_type in (2, 3) and len(dets) > 0 and len(base_yolo_dets) > 0:
        sp_chosen_det = dets[chosen_idx]
        matched_base_det, matched_base_idx, matched_base_iou = match_sp_det_to_base_det(
        sp_chosen_det, base_yolo_dets
    )
    if matched_base_det is not None:
        matched_base_cls_name = matched_base_det.cls_name

    # clip corpus ids to corpus size
    N = corpus_size(paths["object_corpus_dir"])
    
    if sa_type in (1, 3) and N == 0:
        return 1e6
    
    def corpus_id(x: float) -> int:
        idx = int(round(x))
        return min(max(0, idx), N - 1)
    
    ins_id = None
    rep_id = None
    ins_scale = None
    rep_scale = None
    
    if sa_type == 1:
        ins_id = corpus_id(v[Vec.INS_CORPUS_ID])
        ins_scale = float(v[Vec.INS_SCALE])
    elif sa_type == 3:
        rep_id = corpus_id(v[Vec.REP_CORPUS_ID])
        rep_scale = float(v[Vec.REP_SCALE])
        
    
    if sa_type in (2, 3):   # removal or replacement
        if lama is None:
            raise ValueError("lama_inpainter is required for removal/replacement")
    

    final_img, sa_log = apply_sa(
        img_sp=sp_img,
        sa_type=sa_type,
        yolo_dets=dets,
        chosen_idx=chosen_idx,
        sam_segmenter=sam,
        lama_inpainter=lama,
        object_corpus_dir=paths["object_corpus_dir"],
        ins_corpus_id=ins_id,
        ins_scale=ins_scale,
        rep_corpus_id=rep_id,
        rep_scale=rep_scale,
    )
    record["sa"].update(sa_log)
    
    if sa_type in (2, 3):
        record["sa"]["matched_base_det_idx"] = int(matched_base_idx)
        record["sa"]["matched_base_iou"] = float(matched_base_iou)
        record["sa"]["matched_base_cls_name"] = matched_base_cls_name

    tmp_path = out_dir / f"img{image_id}_eval{eval_idx:04d}_sa{sa_type}.png"
    final_img.save(tmp_path)
    trf_caption = vlm_runner.caption(str(tmp_path.resolve()), prompt=cfg["vlm"]["caption_prompt"])
    record["transformed_image_path"] = str(tmp_path.resolve())
    record["transformed_caption"] = trf_caption
    print("Transformed caption:", trf_caption)

    # LLM judge score
    score_1_5, norm, label = judge_score_row(record, llm_cfg)
    record["llm_judge"] = label
    record["llm_score_1_to_5"] = score_1_5
    record["llm_score_norm_0_1"] = norm
    record["sa"]["chosen_idx"] = int(chosen_idx)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # norm = wrongness in [0,1] (0=correct, 1=wrong)
    # GA minimizes -> return -norm so it maximizes wrongness
    return -float(norm)