from __future__ import annotations

from typing import Any, Dict
import numpy as np

from optimization_algorithms.ga_fitness_vlm import vlm_mt_fitness
from optimization_algorithms.particle_swarm_optimization.pso import PSO


def run_pso(cfg, **kwargs_from_main):
    var_bound = np.array(cfg["transformations"]["var_bound"], dtype=float)
    lb = var_bound[:, 0]
    ub = var_bound[:, 1]

    pso_params = cfg["pso"]

    cache = {}  # persistent cache for this optimizer run

    pso_kwargs = dict(kwargs_from_main)
    pso_kwargs["cfg"] = cfg
    pso_kwargs["cache"] = cache

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