"""Tuning stages and the main comparison. Each run appends a row to
results/tune_all.csv, and interrupted stages resume where they stopped.

  python3 tune_all.py --stage budget     evaluation budget
  python3 tune_all.py --stage hidden     hidden units per problem
  python3 tune_all.py --stage shared     s, vmax, init_spread (gbest)
  python3 tune_all.py --stage swarm      s for every algorithm (reported only)
  python3 tune_all.py --stage kfixed     k for random grouping
  python3 tune_all.py --stage coop       s x nr for MCPSO and DCPSO
  python3 tune_all.py --stage lambda     weight decay
  python3 tune_all.py --stage main       final comparison, scored on test
  python3 tune_all.py --stage all        budget .. coop

Shared parameters are tuned once on gbest and reused by all four algorithms,
so differences come from the grouping, not the tuning. Tuning selects on a
validation split; the test split is only used by stage main.
"""

import argparse
import csv
import os

# One BLAS thread per worker (parallelism is across runs). Must be set before
# numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import collections  # noqa: E402
import functools  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
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
    out = [rng.permutation(np.concatenate(p)) for p in (tr, va, te)]
    return out[0], out[1], out[2]


def measures(w, X, T, n_in, n_hid, n_out, kind):
    """Scores for one network. "primary" is macro-F1 for classification and
    -RMSE for regression, so larger is always better."""
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
            "primary": float(np.mean(f1s)),      # macro-F1 drives selection
            "accuracy": float((pred == true).mean()),
            "macro_f1": float(np.mean(f1s)),
            "min_recall": float(np.min(recalls)),
        }

    rmse = float(np.sqrt(np.mean((Y - T) ** 2)))
    var = float(np.var(T))
    return {
        "primary": -rmse,                        # negative, so larger is better
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
        "selected_nh": cfg.get("selected_nh", ""),
        "iterations": iters, "restructures": restr,
        "expected_restructures": expected,
        "degenerate": bool(cfg["mode"] in (2, 3) and restr < expected),
        # Selection uses val_score only.
        "val_score": vm["primary"], "test_score": tm["primary"],
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
        n = n_weights(problem, TUNE_NH[problem])
        budgets = {int(p * one_pass_evals(n, 10, 2)): f"{p}x" for p in (1, 2, 4, 8, 16)}
        for b in (10000, 200000):
            budgets.setdefault(b, "fixed")
        for budget in sorted(budgets):
            for mode in (0, 1, 2, 3):
                for seed in range(seeds):
                    c = base(problem, mode, seed, "budget")
                    c.update(budget=budget)
                    out.append(c)
    return out


def stage_shared(seeds):
    """s x vmax x init_spread on gbest, all at the s=10 budget so larger
    swarms don't get more evaluations."""
    out = []
    for problem in ALL_PROBLEMS:
        for s in (5, 10, 20, 30):
            for vmax in (0.5, 1.0, 2.0):
                for spread in (0.1, 0.5, 1.0):
                    for seed in range(seeds):
                        c = base(problem, 0, seed, "shared")
                        c.update(s=s, vmax=vmax, init_spread=spread)
                        out.append(c)
    return out


def stage_hidden(seeds):
    """Hidden units per problem, for all four algorithms, under two budgets:
    scaled (per-size budget, same effort per weight) and fixed (the nh=20
    budget for every size)."""
    out = []
    for problem in ALL_PROBLEMS:
        fixed = derived_budget(problem, 20)     # nh=20 budget, used for all sizes
        for n_hid in HIDDEN_UNITS:
            scaled = derived_budget(problem, n_hid)
            for mode in (0, 1, 2, 3):
                for seed in range(seeds):
                    for label, budget in (("scaled", scaled), ("fixed", fixed)):
                        c = base(problem, mode, seed, "hidden")
                        c.update(n_hidden=n_hid, budget=budget,
                                 budget_mode=label)
                        out.append(c)
    return out


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
    out = []
    for problem in ALL_PROBLEMS:
        for n_hid in HIDDEN_UNITS:
            for mode in (0, 1, 2, 3):
                for seed in range(seeds):
                    c = base(problem, mode, seed, "main")
                    c.update(
                        n_hidden=n_hid,
                        budget=derived_budget(problem, n_hid),
                        k_fixed=SETTLED_K[problem],
                        selected_nh=SETTLED_NH[problem],
                    )
                    out.append(c)
    return out


def stage_lambda(seeds):
    """Weight decay on iris (163 weights) and digits (2314), gbest only.

    The penalty is plain lambda * sum(w^2), not dennis2020's normalised form,
    so it grows with the number of weights.
    """
    out = []
    for problem, n_hid in (("iris", 20), ("digits", 32)):
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            for seed in range(seeds):
                c = base(problem, 0, seed, "lambda")
                c.update(n_hidden=n_hid, budget=derived_budget(problem, n_hid))
                c["lambda"] = lam
                out.append(c)
    return out


def stage_swarm(seeds):
    """s for all four algorithms. Reported only; stage_main keeps the gbest
    value."""
    out = []
    for problem in ALL_PROBLEMS:
        for mode in (0, 1, 2, 3):
            for s in (5, 10, 20, 30):
                for seed in range(seeds):
                    c = base(problem, mode, seed, "swarm")
                    c.update(s=s)
                    out.append(c)
    return out


def stage_kfixed(seeds):
    """Number of sub-swarms k for random grouping. Run before stage_main."""
    out = []
    for problem in ALL_PROBLEMS:
        for k in (2, 5, 10, 20, 40):
            for seed in range(seeds):
                c = base(problem, 1, seed, "kfixed")
                c.update(k_fixed=k)
                out.append(c)
    return out


def stage_coop(seeds):
    """nr x s for MCPSO and DCPSO at the s=10, nr=2 budget. Large s with
    small nr may degenerate; see the `degenerate` column."""
    out = []
    for problem in ALL_PROBLEMS:
        for mode in (2, 3):
            for nr in (2, 4, 8, 16):
                for s in (5, 10, 20):
                    for seed in range(seeds):
                        c = base(problem, mode, seed, "coop")
                        c.update(nr=nr, s=s)
                        out.append(c)
    return out


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
        summarise(out)
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
    summarise(out)


def summarise(path):
    """Mean val_score per parameter value, averaged over the others."""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return
    axes = {"budget": ("budget",), "lambda": ("lambda",),
            "shared": ("s", "vmax", "init_spread"),
            "hidden": ("n_hidden",), "swarm": ("s",),
            "main": ("n_hidden",),
            "kfixed": ("k_fixed",), "coop": ("nr", "s")}
    for stage, params in axes.items():
        rs = [r for r in rows if r["stage"] == stage]
        if not rs:
            continue
        print(f"\n=== {stage} ({len(rs)} runs) ===")
        for alg in sorted({r["algorithm"] for r in rs}):
            for param in params:
                g = collections.defaultdict(list)
                for r in rs:
                    if r["algorithm"] == alg:
                        g[r[param]].append(float(r["val_score"]))
                if len(g) < 2:
                    continue
                print(f"  {alg:<10} {param}:", end="")
                for v in sorted(g, key=float):
                    print(f"  {v}={np.mean(g[v]):.4f}", end="")
                print()


if __name__ == "__main__":
    main()
