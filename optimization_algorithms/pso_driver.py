from __future__ import annotations

import numpy as np
from PIL import Image

from optimization_algorithms.ga_fitness_vlm import vlm_mt_fitness
from optimization_algorithms.particle_swarm_optimization.pso import PSO

from Utils.corpus import corpus_size
from Utils.vector_layout import Vec


def run_pso(cfg, **kwargs_from_main):
    var_bound = np.array(cfg["transformations"]["var_bound"], dtype=float)
    pso_params = cfg["pso"]

    cache = {}  # persistent cache for this optimizer run

    pso_kwargs = dict(kwargs_from_main)
    pso_kwargs["cfg"] = cfg
    pso_kwargs["cache"] = cache

    # ----- dynamic bound for TARGET_DET_IDX -----
    base_image_path = kwargs_from_main["base_image_path"]
    yolo = kwargs_from_main["yolo_detector"]
    topk = int(cfg["thresholds"].get("yolo_topk", 10))

    base_img = Image.open(base_image_path).convert("RGB")
    base_yolo_dets = yolo.detect_topk(base_img, topk=topk)

    if len(base_yolo_dets) == 0:
        raise ValueError("No base-image detections found.")

    cache["base_img"] = base_img
    cache["base_yolo_dets"] = base_yolo_dets

    var_bound[Vec.TARGET_DET_IDX] = [0, len(base_yolo_dets) - 1]

    # ----- dynamic bounds for corpus ids -----
    N = corpus_size(cfg["paths"]["object_corpus_dir"])
    if N == 0:
        raise ValueError("Empty object corpus.")

    var_bound[Vec.INS_CORPUS_ID] = [0, N - 1]
    var_bound[Vec.REP_CORPUS_ID] = [0, N - 1]

    # bounds are computed after dynamic updates
    lb = var_bound[:, 0]
    ub = var_bound[:, 1]

    def f(x: np.ndarray) -> float:
        return float(vlm_mt_fitness(x, **pso_kwargs))

    optimizer = PSO(
        func=f,
        lb=lb,
        ub=ub,
        swarmsize=int(pso_params.get("swarmsize", 10)),
        omega=float(pso_params.get("omega", 0.5)),
        phip=float(pso_params.get("phip", 0.5)),
        phig=float(pso_params.get("phig", 0.5)),
        maxiter=int(pso_params.get("maxiter", 10)),
        minstep=float(pso_params.get("minstep", 1e-8)),
        minfunc=float(pso_params.get("minfunc", 1e-8)),
        debug=bool(pso_params.get("debug", True)),
    )

    best_x, best_f = optimizer.run()
    return {"best_variable": best_x, "best_function": best_f}