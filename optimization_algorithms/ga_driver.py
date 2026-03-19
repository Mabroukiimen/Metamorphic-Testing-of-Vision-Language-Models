# optimization_algorithms/ga_driver.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from optimization_algorithms.genetic_algorithm.geneticalgorithm import geneticalgorithm
from optimization_algorithms.ga_fitness_vlm import vlm_mt_fitness


from PIL import Image
from Utils.vector_layout import Vec

from Utils.corpus import corpus_size

def run_ga(cfg, **kwargs_from_main):
    var_bound = np.array(cfg["transformations"]["var_bound"], dtype=float)
    
    layout = cfg["layout"]
    
    print("Vec.N =", Vec.N)
    print("len(var_bound) =", len(var_bound))
    print("len(layout) =", len(layout))


    base_image_path = kwargs_from_main["base_image_path"]
    yolo = kwargs_from_main["yolo_detector"]
    topk = int(cfg["thresholds"].get("yolo_topk", 10))

    base_img = Image.open(base_image_path).convert("RGB")
    base_yolo_dets = yolo.detect_topk(base_img, topk=topk)

    if len(base_yolo_dets) == 0:
        raise ValueError("No base-image detections found.")

    var_bound[Vec.TARGET_DET_IDX] = [0, len(base_yolo_dets) - 1]
    
    N = corpus_size(cfg["paths"]["object_corpus_dir"])
    if N == 0:
        raise ValueError("Empty object corpus.")
    
    var_bound[Vec.INS_CORPUS_ID] = [0, N - 1]
    var_bound[Vec.REP_CORPUS_ID] = [0, N - 1]

    dim = var_bound.shape[0]
    algo_params = cfg["ga"]

    cache = {}
    cache["base_img"] = base_img
    cache["base_yolo_dets"] = base_yolo_dets

    ga_kwargs = dict(kwargs_from_main)
    ga_kwargs["cfg"] = cfg
    ga_kwargs["cache"] = cache
    
    var_type = np.array([["real"]] * dim, dtype=object)
    

    var_type[Vec.SA_TYPE] = ["int"]
    var_type[Vec.TARGET_DET_IDX] = ["int"]
    var_type[Vec.INS_CORPUS_ID] = ["int"]
    var_type[Vec.REP_CORPUS_ID] = ["int"]

    ga = geneticalgorithm(
        function=vlm_mt_fitness,
        dimension=dim,
        kwargs=ga_kwargs,
        variable_type_mixed=var_type,
        variable_boundaries=var_bound,
        algorithm_parameters=algo_params,
        convergence_curve=True,
        progress_bar=True,
    )
    ga.run()
    return {"best_variable": ga.best_variable, "best_function": ga.best_function}



