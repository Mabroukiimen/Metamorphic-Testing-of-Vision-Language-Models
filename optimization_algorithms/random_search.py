import random
from typing import Any, Dict, List
import numpy as np
from PIL import Image

from Utils.vector_layout import Vec
from optimization_algorithms.ga_fitness_vlm import vlm_mt_fitness


def _sample_gene(low: float, high: float, gene_type: str):
    if gene_type == "int":
        return int(random.randint(int(round(low)), int(round(high))))
    return float(random.uniform(float(low), float(high)))


def _build_vector(var_bound, layout) -> List[float]:
    v = []

    for item in layout:
        idx = int(item["index"])
        gene_type = item["type"]
        low, high = var_bound[idx]

        v.append(_sample_gene(low, high, gene_type))

    return v


def run_random_search(cfg, **kwargs_from_main):
    var_bound = np.array(cfg["transformations"]["var_bound"], dtype=float)
    layout = cfg["layout"]
    rs_cfg = cfg.get("random_search", {})

    base_image_path = kwargs_from_main["base_image_path"]
    yolo = kwargs_from_main["yolo_detector"]
    topk = int(cfg["thresholds"].get("yolo_topk", 10))

    # same preparation as GA
    base_img = Image.open(base_image_path).convert("RGB")
    base_yolo_dets = yolo.detect_topk(base_img, topk=topk)

    if len(base_yolo_dets) == 0:
        raise ValueError("No base-image detections found.")

    # dynamic bound for target_det_idx
    var_bound[Vec.TARGET_DET_IDX] = [0, len(base_yolo_dets) - 1]

    num_samples = int(rs_cfg.get("num_samples", 100))
    seed = rs_cfg.get("seed", 42)
    random.seed(seed)

    cache = {}
    cache["base_img"] = base_img
    cache["base_yolo_dets"] = base_yolo_dets

    rs_kwargs = dict(kwargs_from_main)
    rs_kwargs["cfg"] = cfg
    rs_kwargs["cache"] = cache

    best_vector = None
    best_fitness = float("inf")
    history = []

    for i in range(num_samples):
        v = _build_vector(var_bound, layout)
        f = float(vlm_mt_fitness(v, **rs_kwargs))

        history.append({
            "iter": i,
            "fitness": float(f),
            "vector": list(map(float, v)),
        })

        print(f"[RANDOM] iter={i} vector={v}")
        print(f"[RANDOM] iter={i} fitness={f}")

        if f < best_fitness:
            best_fitness = f
            best_vector = v
            print(f"[RANDOM] iter={i} new best fitness={best_fitness}")

    return {
        "best_variable": best_vector,
        "best_function": best_fitness,
        "history": history,
    }