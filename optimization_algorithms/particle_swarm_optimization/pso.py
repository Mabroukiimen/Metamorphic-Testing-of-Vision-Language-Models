import numpy as np


class PSO:
    def __init__(
        self,
        func,
        lb,
        ub,
        swarmsize=100,
        omega=0.5,
        phip=0.5,
        phig=0.5,
        maxiter=100,
        minstep=1e-8,
        minfunc=1e-8,
        debug=False,
    ):
        self.func = func
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)

        assert self.lb.shape == self.ub.shape  #Lower- and upper-bounds must be the same length
        assert np.all(self.ub > self.lb)       #upper-bounds must be greater than lower-bounds

        self.swarmsize = int(swarmsize)
        self.omega = float(omega)
        self.phip = float(phip)
        self.phig = float(phig)
        self.maxiter = int(maxiter)
        self.minstep = float(minstep)
        self.minfunc = float(minfunc)
        self.debug = bool(debug)

        self.dim = len(self.lb)

    def run(self):
        # initialize particle positions uniformly in bounds
        x = np.random.uniform(self.lb, self.ub, size=(self.swarmsize, self.dim))

        # initialize velocities
        vhigh = np.abs(self.ub - self.lb)
        vlow = -vhigh
        v = np.random.uniform(vlow, vhigh, size=(self.swarmsize, self.dim))

        # personal bests
        p = x.copy()
        fp = np.array([self.func(xi) for xi in x], dtype=float)

        # global best
        g_idx = np.argmin(fp)
        g = p[g_idx].copy()
        fg = fp[g_idx]

        if self.debug:
            print(f"[PSO] init best_f = {fg}")

        for it in range(self.maxiter):
            rp = np.random.uniform(size=(self.swarmsize, self.dim))
            rg = np.random.uniform(size=(self.swarmsize, self.dim))

            # update velocities and positions
            v = (
                self.omega * v
                + self.phip * rp * (p - x)
                + self.phig * rg * (g - x)
            )
            x = x + v

            # clip to bounds
            x = np.clip(x, self.lb, self.ub)

            # evaluate
            fx = np.array([self.func(xi) for xi in x], dtype=float)

            # update personal bests
            improved = fx < fp
            p[improved] = x[improved]
            fp[improved] = fx[improved]

            # update global best
            g_idx_new = np.argmin(fp)
            g_new = p[g_idx_new].copy()
            fg_new = fp[g_idx_new]

            step = np.linalg.norm(g_new - g)
            fchange = abs(fg - fg_new)

            g = g_new
            fg = fg_new

            if self.debug:
                print(f"[PSO] iter={it+1}/{self.maxiter} best_f={fg} step={step:.6g} fchange={fchange:.6g}")

            if step <= self.minstep:
                if self.debug:
                    print("[PSO] stopping: minstep reached")
                break

            if fchange <= self.minfunc:
                if self.debug:
                    print("[PSO] stopping: minfunc reached")
                break

        return g, fg