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
    sa_type = int(round(v[Vec.SA_TYPE]))  # 0..3
    
    #sa_type = 0  # TEMP: run SP-only (no YOLO/SAM/LLM judge)

    # prepare logging shell
    eval_idx = cache.get("eval_idx", 0)
    cache["eval_idx"] = eval_idx + 1

    record: Dict[str, Any] = {
        "image": base_image_path,
        "image_id": image_id,
        "vector": list(map(float, v)),
        "base_caption": base_caption,
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
        print("transformed caption", trf_caption)
        dist = sts.distance(base_caption, trf_caption)
        record["transformed_image_path"] = str(tmp_path)
        record["transformed_caption"] = trf_caption
        record["sts_distance"] = float(dist)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # minimize (-dist) -> maximize dist
        return -float(dist)

    # ---- SA path ----
    topk = int(thr.get("yolo_topk", 10))
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

    # clip corpus ids to corpus size
    N = corpus_size(paths["object_corpus_dir"])
    def corpus_id(x: float) -> int:
        idx = int(round(x))
        if not (0 <= idx < N):
            raise ValueError(f"Corpus index out of range: {idx}, corpus size={N}")
        return idx
    
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
        
    lama = kwargs.get("lama_inpainter", None)

    if sa_type == 2:   # removal
        if lama is None:
            raise ValueError("lama_inpainter is required for removal")
    # use lama here


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

    tmp_path = out_dir / f"img{image_id}_eval{eval_idx:04d}_sa{sa_type}.png"
    final_img.save(tmp_path)
    trf_caption = vlm_runner.caption(str(tmp_path), prompt=cfg["vlm"]["caption_prompt"])
    record["transformed_image_path"] = str(tmp_path)
    record["transformed_caption"] = trf_caption

    # LLM judge score
    score_1_5, norm, label = judge_score_row(record, llm_cfg)
    record["llm_judge"] = label
    record["llm_score_1_to_5"] = score_1_5
    record["llm_score_norm_0_1"] = norm

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # norm = wrongness in [0,1] (0=correct, 1=wrong)
    # GA minimizes -> return -norm so it maximizes wrongness
    return -float(norm)