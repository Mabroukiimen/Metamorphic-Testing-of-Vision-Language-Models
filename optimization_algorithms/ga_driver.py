# optimization_algorithms/ga_driver.py
from __future__ import annotations

from typing import Any, Dict
import numpy as np

from optimization_algorithms.genetic_algorithm.geneticalgorithm import geneticalgorithm
from optimization_algorithms.ga_fitness_vlm import vlm_mt_fitness


def run_ga(cfg, **kwargs_from_main):
    var_bound = np.array(cfg["transformations"]["var_bound"], dtype=float)
    dim = var_bound.shape[0]
    algo_params = cfg["ga"]

    cache = {}  # <--- persistent cache for this GA run

    ga_kwargs = dict(kwargs_from_main)
    ga_kwargs["cfg"] = cfg
    ga_kwargs["cache"] = cache

    ga = geneticalgorithm(
        function=vlm_mt_fitness,
        dimension=dim,
        kwargs=ga_kwargs,
        variable_type="real",
        variable_boundaries=var_bound,
        algorithm_parameters=algo_params,
        convergence_curve=True,
        progress_bar=True,
    )

    ga.run()
    return {"best_variable": ga.best_variable, "best_function": ga.best_function}

