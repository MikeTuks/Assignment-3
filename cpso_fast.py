"""PSO and cooperative PSO trainers for a one-hidden-layer network.

  mode 0  gbest PSO (baseline)
  mode 1  random-grouping CPSO, regroups every iteration (dennis2020)
  mode 2  MCPSO, merges sub-swarms: CPSO-S -> PSO (douglas2018 Alg. 4)
  mode 3  DCPSO, splits sub-swarms: PSO -> CPSO-S (douglas2018 Alg. 3)

All minimise f(x) + lambda*WD and stop after a fixed number of evaluations.
Sub-swarm j owns dims[bounds[j]:bounds[j+1]] (CSR-style, since Numba can't
hold ragged lists). State is stored as (s, n) arrays.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _unpack(w, n_in, n_hid, n_out):
    """Flat weights -> W1 (n_hid, n_in), b1, W2 (n_out, n_hid), b2."""
    i = n_hid * n_in
    W1 = w[:i].reshape(n_hid, n_in)
    b1 = w[i:i + n_hid]
    i += n_hid
    W2 = w[i:i + n_out * n_hid].reshape(n_out, n_hid)
    i += n_out * n_hid
    b2 = w[i:i + n_out]
    return W1, b1, W2, b2


def forward(w, X, n_in, n_hid, n_out):
    """Network output."""
    W1, b1, W2, b2 = _unpack(w, n_in, n_hid, n_out)
    return np.tanh(X @ W1.T + b1) @ W2.T + b2


@njit(cache=True, fastmath=True)
def fitness_fast(w, X, T, n_in, n_hid, n_out, decay):
    """Half MSE + decay * sum(w^2). One call = one evaluation.

    tanh(x) = 1 - 2/(exp(2x) + 1): matches np.tanh to 1e-16, about 2x
    faster, and still gives exactly +/-1 when exp overflows.
    """
    W1, b1, W2, b2 = _unpack(w, n_in, n_hid, n_out)
    Z = X @ W1.T + b1
    H = 1.0 - 2.0 / (np.exp(2.0 * Z) + 1.0)
    Y = H @ W2.T + b2
    d = Y - T
    e = 0.5 * np.sum(d * d) / X.shape[0]
    if decay > 0.0:
        e += decay * np.sum(w * w)
    return e


@njit(cache=True)
def eval_context_flat(context, pos, i, lo, hi, dims, X, T,
                      n_in, n_hid, n_out, decay):
    """f(b(j, z)): context with dims[lo:hi] set to particle i. Restores context."""
    saved = np.empty(hi - lo)
    for t in range(hi - lo):
        d = dims[lo + t]
        saved[t] = context[d]
        context[d] = pos[i, lo + t]
    val = fitness_fast(context, X, T, n_in, n_hid, n_out, decay)
    for t in range(hi - lo):
        context[dims[lo + t]] = saved[t]
    return val


@njit(cache=True)
def reseed_flat(pos, vel, pbest, pbest_val, gbest, gbest_val,
                context, dims, bounds, k, s, spread, seed_vel):
    """After regrouping: gbest = context slice, particles scattered around it."""
    for j in range(k):
        lo, hi = bounds[j], bounds[j + 1]
        gbest_val[j] = 1e300
        for t in range(lo, hi):
            gbest[t] = context[dims[t]]
        for i in range(s):
            pbest_val[j, i] = 1e300
            for t in range(lo, hi):
                base = context[dims[t]]
                pos[i, t] = base + spread * (np.random.random() * 2.0 - 1.0)
                pbest[i, t] = pos[i, t]
                vel[i, t] = seed_vel * (np.random.random() * 2.0 - 1.0)


@njit(cache=True)
def split_bounds(bounds, k, nr, n):
    """DCPSO: split each sub-swarm into nr parts, in place. Returns new k."""
    new_bounds = np.empty(n + 1, dtype=np.int64)
    new_bounds[0] = 0
    m = 0
    for j in range(k):
        lo, hi = bounds[j], bounds[j + 1]
        d = hi - lo
        if d <= 1:                      # 1-D sub-swarms can't split
            m += 1
            new_bounds[m] = hi
            continue
        parts = min(nr, d)
        base, rem = d // parts, d % parts
        cur = lo
        for q in range(parts):
            cur += base + (1 if q < rem else 0)
            m += 1
            new_bounds[m] = cur
    bounds[:m + 1] = new_bounds[:m + 1]
    return m


@njit(cache=True)
def merge_bounds(bounds, k, nr):
    """MCPSO: merge every nr neighbouring sub-swarms, in place. Returns new k."""
    new_bounds = np.empty(k + 1, dtype=np.int64)
    new_bounds[0] = 0
    m = 0
    j = 0
    while j < k:
        j = min(j + nr, k)
        m += 1
        new_bounds[m] = bounds[j]
    bounds[:m + 1] = new_bounds[:m + 1]
    return m


@njit(cache=True)
def train_fast(mode, n, X, T, n_in, n_hid, n_out, decay,
               s, budget, omega, c1, c2, vmax, init_spread,
               k_fixed, nr, seed):
    """Returns (weights, fitness, evals, iterations, restructures)."""
    np.random.seed(seed)

    dims = np.arange(n)
    bounds = np.empty(n + 1, dtype=np.int64)
    bounds[0] = 0
    if mode == 1:                       # k_fixed equal sub-swarms
        k = min(k_fixed, n)
        base, rem = n // k, n % k
        cur = 0
        for j in range(k):
            cur += base + (1 if j < rem else 0)
            bounds[j + 1] = cur
    elif mode == 2:                     # n one-dimensional sub-swarms
        k = n
        for j in range(n + 1):
            bounds[j] = j
    else:                               # one swarm over all dims
        k = 1
        bounds[1] = n

    pos = np.zeros((s, n))
    vel = np.zeros((s, n))
    pbest = np.zeros((s, n))
    pbest_val = np.full((n, s), 1e300)
    gbest = np.zeros(n)
    gbest_val = np.full(n, 1e300)

    context = np.empty(n)
    for i in range(n):
        context[i] = init_spread * (np.random.random() * 2.0 - 1.0)

    reseed_flat(pos, vel, pbest, pbest_val, gbest, gbest_val,
                context, dims, bounds, k, s, init_spread, 0.1 * init_spread)

    # Iterations between restructurings, so each grouping gets an equal share
    # of evaluations: budget / (s * sum of k over the groupings). Adapted from
    # douglas2018 eq. 3, which splits iterations instead. analyse.schedule
    # replays this.
    n_f = 1
    if mode >= 2:
        m = max(1, int(np.log(n) / np.log(nr)))
        total_k = 0.0
        kk = 1.0 if mode == 3 else float(n)
        for _ in range(m + 1):
            total_k += kk
            kk = kk * nr if mode == 3 else kk / nr
        n_f = max(1, int(budget / (s * total_k)))

    evals = 0
    it = 0
    n_restructures = 0

    while evals < budget:
        it += 1
        for j in range(k):
            lo, hi = bounds[j], bounds[j + 1]
            for i in range(s):
                if evals >= budget:
                    break
                val = eval_context_flat(context, pos, i, lo, hi, dims,
                                        X, T, n_in, n_hid, n_out, decay)
                evals += 1
                if val < pbest_val[j, i]:
                    pbest_val[j, i] = val
                    for t in range(lo, hi):
                        pbest[i, t] = pos[i, t]
                    if val < gbest_val[j]:
                        gbest_val[j] = val
                        for t in range(lo, hi):
                            gbest[t] = pos[i, t]
                            context[dims[t]] = pos[i, t]
            for i in range(s):
                for t in range(lo, hi):
                    r1 = np.random.random()
                    r2 = np.random.random()
                    v = (omega * vel[i, t]
                         + c1 * r1 * (pbest[i, t] - pos[i, t])
                         + c2 * r2 * (gbest[t] - pos[i, t]))
                    if v > vmax:
                        v = vmax
                    elif v < -vmax:
                        v = -vmax
                    vel[i, t] = v
                    pos[i, t] += v

        regrouped = False
        if mode == 1:                   # shuffle dims (Fisher-Yates)
            for i in range(n - 1, 0, -1):
                r = np.random.randint(0, i + 1)
                tmp = dims[i]
                dims[i] = dims[r]
                dims[r] = tmp
            regrouped = True
        elif mode == 2 and it % n_f == 0 and k > 1:
            k = merge_bounds(bounds, k, nr)
            n_restructures += 1
            regrouped = True
        elif mode == 3 and it % n_f == 0 and k < n:
            k = split_bounds(bounds, k, nr, n)
            n_restructures += 1
            regrouped = True
        if regrouped:
            reseed_flat(pos, vel, pbest, pbest_val, gbest, gbest_val,
                        context, dims, bounds, k, s,
                        init_spread * 0.1, 0.01 * init_spread)

    best = fitness_fast(context, X, T, n_in, n_hid, n_out, decay)
    return context, best, evals, it, n_restructures


if __name__ == "__main__":
    # Merge and split must keep [0, n) partitioned with no empty sub-swarms.
    n = 12
    bounds = np.arange(n + 1).astype(np.int64)
    k = n
    while k > 1:
        k = merge_bounds(bounds, k, 3)
        assert bounds[0] == 0 and bounds[k] == n, "merge lost dimensions"
        assert (np.diff(bounds[:k + 1]) > 0).all(), "merge made an empty sub-swarm"
    bounds[1] = n
    for _ in range(5):
        k = split_bounds(bounds, k, 3, n)
        assert bounds[0] == 0 and bounds[k] == n, "split lost dimensions"
        assert (np.diff(bounds[:k + 1]) > 0).all(), "split made an empty sub-swarm"
    print("merge/split bounds check passed.")
