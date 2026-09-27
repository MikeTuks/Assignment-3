"""Tuning stages and the main comparison; results go to results/tune_all.csv.

Shared parameters are tuned once on gbest and reused by all four algorithms.
Tuning selects on a validation split; only stage main uses the test split.
"""

import argparse
import csv
import os

# One BLAS thread per worker (parallelism is across runs). Must be set before
# numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import functools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from itertools import product  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent / "data"))
import datasets  # noqa: E402
from cpso_fast import forward, train_fast  # noqa: E402

RESULTS = Path(__file__).parent / "results"

ALGORITHMS = {0: "gbest", 1: "randgroup", 2: "mcpso", 3: "dcpso"}
ALL_PROBLEMS = ("iris", "ionosphere", "sonar", "digits",
                "sine", "concrete", "friedman", "communities")
HIDDEN_UNITS = (4, 10, 20, 32)

# Hidden units used while tuning (stage_hidden picks the final ones).
TUNE_NH = {
    "iris": 20, "ionosphere": 20, "sonar": 20, "digits": 32,
    "sine": 20, "concrete": 20, "friedman": 20, "communities": 20,
}

# Budget = 8 passes through the grouping schedule, about 16 * n * s, so it is
# linear in the number of weights. At 8 passes no run degenerated.
BUDGET_PASSES = 8


@functools.cache
def n_weights(problem, n_hid):
    X, T, _ = datasets.load(problem)
    return datasets.n_weights(X.shape[1], n_hid, T.shape[1])


def derived_budget(problem, n_hid, s=10, nr=2):
    """Evaluation budget for one problem at BUDGET_PASSES passes."""
    return int(BUDGET_PASSES * one_pass_evals(n_weights(problem, n_hid), s, nr))


def split_three(X, T, seed, train=0.6, val=0.2, kind="regression"):
    """60/20/20 train/val/test indices, stratified for classification."""
    rng = np.random.default_rng(seed)

    if kind != "classification":
        idx = rng.permutation(len(X))
        a, b = int(train * len(X)), int((train + val) * len(X))
        return idx[:a], idx[a:b], idx[b:]

    tr, va, te = [], [], []
    for c in range(T.shape[1]):
        members = rng.permutation(np.flatnonzero(T[:, c] == 1.0))
        a, b = int(train * len(members)), int((train + val) * len(members))
        tr.append(members[:a])
        va.append(members[a:b])
        te.append(members[b:])
    # Shuffle so the splits aren't ordered by class.
    return [rng.permutation(np.concatenate(p)) for p in (tr, va, te)]


def measures(w, X, T, n_in, n_hid, n_out, kind):
    """Scores for one network."""
    Y = forward(w, X, n_in, n_hid, n_out)

    if kind == "classification":
        pred, true = Y.argmax(axis=1), T.argmax(axis=1)
        f1s, recalls = [], []
        for c in range(T.shape[1]):
            tp = float(np.sum((pred == c) & (true == c)))
            fp = float(np.sum((pred == c) & (true != c)))
            fn = float(np.sum((pred != c) & (true == c)))
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
            recalls.append(rec)
        return {
            "accuracy": float((pred == true).mean()),
            "macro_f1": float(np.mean(f1s)),
            "min_recall": float(np.min(recalls)),
        }

    rmse = float(np.sqrt(np.mean((Y - T) ** 2)))
    var = float(np.var(T))
    return {
        "rmse": rmse,
        "r2": float(1.0 - np.mean((Y - T) ** 2) / var) if var > 0 else 0.0,
        "min_recall": float("nan"),
    }


def run_one(cfg):
    """Train and score one network."""
    t0 = time.time()
    problem = cfg["problem"]
    X, T, kind = datasets.load(problem)
    n_hid = cfg["n_hidden"]
    n_in, n_out = X.shape[1], T.shape[1]
    n = datasets.n_weights(n_in, n_hid, n_out)

    tr, va, te = split_three(X, T, cfg["seed"], kind=kind)

    w, fit, evals, iters, restr = train_fast(
        cfg["mode"], n, X[tr], T[tr], n_in, n_hid, n_out, cfg["lambda"],
        cfg["s"], cfg["budget"], cfg["omega"], cfg["c1"], cfg["c2"],
        cfg["vmax"], cfg["init_spread"], cfg["k_fixed"], cfg["nr"],
        cfg["seed"],
    )

    vm = measures(w, X[va], T[va], n_in, n_hid, n_out, kind)
    tm = measures(w, X[te], T[te], n_in, n_hid, n_out, kind)

    # Degenerate = ran out of budget before finishing the schedule.
    expected = int(np.log(n) / np.log(cfg["nr"])) if cfg["mode"] in (2, 3) else 0
    return {
        "stage": cfg["stage"], "problem": problem, "kind": kind,
        "algorithm": ALGORITHMS[cfg["mode"]], "n_hidden": n_hid,
        "n_weights": n, "seed": cfg["seed"], "s": cfg["s"], "nr": cfg["nr"],
        "k_fixed": cfg["k_fixed"], "vmax": cfg["vmax"],
        "init_spread": cfg["init_spread"], "omega": cfg["omega"],
        "lambda": cfg["lambda"], "budget": cfg["budget"],
        "budget_mode": cfg["budget_mode"],
        "selected_nh": cfg["selected_nh"],
        "iterations": iters, "restructures": restr,
        "expected_restructures": expected,
        "degenerate": bool(cfg["mode"] in (2, 3) and restr < expected),
        # Selection uses val_score only. Macro-F1 or -RMSE, so larger is better.
        "val_score": vm["macro_f1"] if kind == "classification" else -vm["rmse"],
        "test_score": tm["macro_f1"] if kind == "classification" else -tm["rmse"],
        "val_accuracy": vm.get("accuracy", ""),
        "test_accuracy": tm.get("accuracy", ""),
        "test_macro_f1": tm.get("macro_f1", ""),
        "test_min_recall": tm["min_recall"],
        "test_rmse": tm.get("rmse", ""), "test_r2": tm.get("r2", ""),
        "train_fitness": fit, "evals_used": evals,
        "seconds": round(time.time() - t0, 1),
    }


def base(problem, mode, seed, stage):
    return {
        "problem": problem, "mode": mode, "seed": seed, "stage": stage,
        "n_hidden": TUNE_NH[problem],
        "budget": derived_budget(problem, TUNE_NH[problem]),
        "omega": 0.72, "c1": 1.49, "c2": 1.49, "lambda": 0.001,
        "s": 10, "vmax": 1.0, "init_spread": 0.1, "k_fixed": 10, "nr": 2,
        "budget_mode": "scaled",        # only stage_hidden changes this
        "selected_nh": "",              # set by stage_main
    }


def one_pass_evals(n, s, nr):
    """Evaluations for one iteration in every grouping: s * sum(k)."""
    m = max(1, int(np.log(n) / np.log(nr)))
    total_k, k = 0.0, float(n)
    for _ in range(m + 1):
        total_k += k
        k /= nr
    return total_k * s


def stage_budget(seeds):
    """Budgets of 1-16 schedule passes per problem, plus fixed 10k and 200k.
    A fixed budget doesn't transfer: 10,000 evaluations is 0.22 passes on
    digits."""
    out = []
    for problem in ALL_PROBLEMS:
        one = one_pass_evals(n_weights(problem, TUNE_NH[problem]), 10, 2)
        budgets = {int(p * one) for p in (1, 2, 4, 8, 16)} | {10000, 200000}
        out += [{**base(problem, mode, seed, "budget"), "budget": b}
                for b, mode, seed in product(sorted(budgets), range(4), range(seeds))]
    return out


def stage_shared(seeds):
    """s x vmax x init_spread on gbest, all at the s=10 budget so larger
    swarms don't get more evaluations."""
    return [{**base(p, 0, seed, "shared"), "s": s, "vmax": v, "init_spread": sp}
            for p, s, v, sp, seed in product(ALL_PROBLEMS, (5, 10, 20, 30),
                                             (0.5, 1.0, 2.0), (0.1, 0.5, 1.0),
                                             range(seeds))]


def stage_hidden(seeds):
    """Hidden units per problem, for all four algorithms, under two budgets:
    scaled (per-size budget, same effort per weight) and fixed (the nh=20
    budget for every size)."""
    return [{**base(p, mode, seed, "hidden"), "n_hidden": nh,
             "budget": derived_budget(p, nh if label == "scaled" else 20),
             "budget_mode": label}
            for p, nh, mode, seed, label in product(ALL_PROBLEMS, HIDDEN_UNITS,
                                                    range(4), range(seeds),
                                                    ("scaled", "fixed"))]


# Picked by the tuning stages (best validation score).
SETTLED_K = {            # stage kfixed, random grouping only
    "iris": 20, "ionosphere": 5, "sonar": 40, "digits": 40,
    "sine": 2, "concrete": 2, "friedman": 2, "communities": 5,
}
SETTLED_NH = {           # stage hidden, fixed-budget reading
    "iris": 20, "ionosphere": 10, "sonar": 10, "digits": 32,
    "sine": 20, "concrete": 10, "friedman": 10, "communities": 20,
}


def stage_main(seeds):
    """Final comparison: 30 seeds, all four sizes, tuned parameters, scored
    on test."""
    return [{**base(p, mode, seed, "main"), "n_hidden": nh,
             "budget": derived_budget(p, nh), "k_fixed": SETTLED_K[p],
             "selected_nh": SETTLED_NH[p]}
            for p, nh, mode, seed in product(ALL_PROBLEMS, HIDDEN_UNITS,
                                             range(4), range(seeds))]


def stage_lambda(seeds):
    """Weight decay on iris (163 weights) and digits (2314), gbest only.

    The penalty is plain lambda * sum(w^2), not dennis2020's normalised form,
    so it grows with the number of weights.
    """
    return [{**base(p, 0, seed, "lambda"), "n_hidden": nh,
             "budget": derived_budget(p, nh), "lambda": lam}
            for (p, nh), lam, seed in product((("iris", 20), ("digits", 32)),
                                              (1e-4, 1e-3, 1e-2, 1e-1),
                                              range(seeds))]


def stage_swarm(seeds):
    """s for all four algorithms. Reported only; stage_main keeps the gbest
    value."""
    return [{**base(p, mode, seed, "swarm"), "s": s}
            for p, mode, s, seed in product(ALL_PROBLEMS, range(4),
                                            (5, 10, 20, 30), range(seeds))]


def stage_kfixed(seeds):
    """Number of sub-swarms k for random grouping. Run before stage_main."""
    return [{**base(p, 1, seed, "kfixed"), "k_fixed": k}
            for p, k, seed in product(ALL_PROBLEMS, (2, 5, 10, 20, 40),
                                      range(seeds))]


def stage_coop(seeds):
    """nr x s for MCPSO and DCPSO at the s=10, nr=2 budget. Large s with
    small nr may degenerate; see the `degenerate` column."""
    return [{**base(p, mode, seed, "coop"), "nr": nr, "s": s}
            for p, mode, nr, s, seed in product(ALL_PROBLEMS, (2, 3),
                                                (2, 4, 8, 16), (5, 10, 20),
                                                range(seeds))]


STAGES = {"budget": stage_budget, "hidden": stage_hidden,
          "shared": stage_shared, "swarm": stage_swarm,
          "kfixed": stage_kfixed, "coop": stage_coop,
          "lambda": stage_lambda, "main": stage_main}

FIELDS = ["stage", "problem", "kind", "algorithm", "n_hidden", "n_weights",
          "seed", "s", "nr", "k_fixed", "vmax", "init_spread", "omega",
          "lambda", "budget", "budget_mode", "selected_nh",
          "iterations", "restructures",
          "expected_restructures", "degenerate", "val_score", "test_score",
          "val_accuracy", "test_accuracy", "test_macro_f1", "test_min_recall",
          "test_rmse", "test_r2", "train_fitness", "evals_used", "seconds"]

# Fields that identify a run, for resuming.
KEY = ("stage", "problem", "algorithm", "n_hidden", "seed", "s", "nr",
       "k_fixed", "vmax", "init_spread", "budget", "budget_mode", "lambda")


def run_key(row):
    return tuple(str(row[k]) for k in KEY)


def done_keys(path):
    if not path.exists():
        return set()
    with open(path) as f:
        return {run_key(r) for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", default="all",
                   choices=(*STAGES, "all"))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=18)
    args = p.parse_args()

    # Budget first: the other stages use its result.
    stages = (("budget", "hidden", "shared", "swarm", "kfixed", "coop")
              if args.stage == "all" else (args.stage,))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "tune_all.csv"
    already = done_keys(out)

    tasks = []
    for st in stages:
        tasks += STAGES[st](args.seeds)
    # Rows store `algorithm`, tasks store `mode`.
    todo = [c for c in tasks
            if run_key({**c, "algorithm": ALGORITHMS[c["mode"]]}) not in already]

    print(f"stages {stages}: {len(tasks)} runs, {len(tasks) - len(todo)} done, "
          f"{len(todo)} to run on {args.workers} workers", flush=True)
    if not todo:
        return

    t0 = time.time()
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_one, c) for c in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                w.writerow(fut.result())
                f.flush()
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el/60:.1f} min, "
                          f"eta {(len(todo)-i)*el/i/60:.1f} min", flush=True)

    print(f"\nwrote {out} in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
