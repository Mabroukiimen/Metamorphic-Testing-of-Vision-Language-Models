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

    # Fitness wrapper: GA passes numpy array x
    def f(x: np.ndarray) -> float:
        return float(vlm_mt_fitness(x, cfg=cfg, **kwargs))

    # vector contains mixed int/float genes, but the library can still operate in 'real'
    # We will round discrete ones inside fitness/decoders (b_* flags, sa_type, indices).
    model = ga(
        function=f,
        dimension=dim,
        variable_type="real",
        variable_boundaries=var_bound,
        algorithm_parameters=algorithm_parameters,
    )

    model.run()

    # model.output_dict typically contains best solution and function value
    out = getattr(model, "output_dict", None)
    if out is None:
        # fallback
        out = {
            "variable": getattr(model, "best_variable", None),
            "function": getattr(model, "best_function", None),
        }

    return out