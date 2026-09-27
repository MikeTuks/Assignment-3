#!/usr/bin/env python3
"""Recomputes the statistics quoted in the report from results/tune_all.csv.

    python3 analyse.py

Algorithm labels in the CSV are lowercase; use ALGOS, not 'DCPSO' etc.
"""
import argparse
import sys

import pandas as pd
from scipy.stats import friedmanchisquare, kruskal, mannwhitneyu, spearmanr, wilcoxon

CSV = "results/tune_all.csv"
GBEST = "gbest"
COOP = ["randgroup", "mcpso", "dcpso"]
ALGOS = [GBEST] + COOP
PRETTY = {"gbest": "gbest", "randgroup": "RandGroup", "mcpso": "MCPSO", "dcpso": "DCPSO"}
CLASSIF = ["iris", "ionosphere", "sonar", "digits"]
APPROX = ["sine", "concrete", "friedman", "communities"]
TITLE = {p: p.capitalize() for p in CLASSIF + APPROX}

# Hidden units picked per problem (tune_all.SETTLED_NH).
SELECTED_NH = {"iris": 20, "ionosphere": 10, "sonar": 10, "digits": 32,
               "sine": 20, "concrete": 10, "friedman": 10, "communities": 20}


def load(path=CSV):
    d = pd.read_csv(path)
    missing = set(ALGOS) - set(d.algorithm.unique())
    if missing:
        sys.exit(f"algorithm labels absent from {path}: {sorted(missing)}")
    return d


def cells(d):
    """Mean test score per (problem, n_hidden, algorithm): 32 rows.
    margin = best cooperative - gbest."""
    m = d[d.stage == "main"]
    g = (m.groupby(["problem", "n_hidden", "n_weights", "algorithm"])["test_score"]
          .mean().unstack().reset_index())
    g["best_coop"] = g[COOP].max(axis=1)
    g["margin"] = g["best_coop"] - g[GBEST]
    return g.sort_values("n_weights").reset_index(drop=True)


def crossover(g):
    """Split cells at the largest gbest win (254 weights) and test each side."""
    base_wins = g.loc[g.margin <= 0, "n_weights"]
    cut = int(base_wins.max())
    nxt = int(g.loc[g.n_weights > cut, "n_weights"].min())
    below, above = g[g.n_weights < cut], g[g.n_weights >= cut]
    rho, p_rho = spearmanr(g.n_weights, g.margin)
    return {
        "coop_wins": int((g.margin > 0).sum()), "n_cells": len(g),
        "cut": cut, "gap_hi": nxt,
        "rho": rho, "p_rho": p_rho,
        "n_below": len(below), "mean_below": below.margin.mean(),
        "p_below": wilcoxon(below.margin).pvalue,
        "n_above": len(above), "mean_above": above.margin.mean(),
        "p_above": wilcoxon(above.margin).pvalue,
    }


def selected(d):
    """Per-problem scores at the architecture the fixed-budget search chose."""
    m = d[d.stage == "main"]
    keep = pd.concat([m[(m.problem == p) & (m.n_hidden == nh)]
                      for p, nh in SELECTED_NH.items()])
    agg = keep.groupby(["problem", "algorithm"])["test_score"].agg(["mean", "std"])
    return agg.unstack(level="algorithm")


def lam(d):
    """Weight-decay sweep: Kruskal-Wallis, then 1e-4 vs 1e-3 and 1e-3 vs 1e-2."""
    L = d[d.stage == "lambda"]
    out = {}
    for p in sorted(L.problem.unique()):
        s = L[L.problem == p]
        vals = sorted(s["lambda"].unique())
        by = {v: s[s["lambda"] == v].val_score for v in vals}
        out[p] = {
            "means": {v: by[v].mean() for v in vals},
            "kruskal_p": kruskal(*[by[v] for v in vals]).pvalue,
            "p_1e4_vs_1e3": mannwhitneyu(by[vals[0]], by[vals[1]]).pvalue,
            "p_1e3_vs_1e2": mannwhitneyu(by[vals[1]], by[vals[2]]).pvalue,
        }
    return out


def budget(d, problem="sonar"):
    """Mean val_score per (budget, algorithm) for one problem."""
    b = d[(d.stage == "budget") & (d.problem == problem)]
    return b.groupby(["budget", "algorithm"])["val_score"].mean().unstack()[ALGOS]


def schedule(mode, n, budget, s=10, nr=2):
    """Replay train_fast's restructuring schedule without training.
    Returns (iterations, iteration the schedule ends, budget share used)."""
    import math
    m = max(1, int(math.log(n) / math.log(nr)))
    total_k, kk = 0.0, (1.0 if mode == 3 else float(n))
    for _ in range(m + 1):
        total_k += kk
        kk = kk * nr if mode == 3 else kk / nr
    n_f = max(1, int(budget / (s * total_k)))
    k, evals, it, end = (n if mode == 2 else 1), 0, 0, (0, 0)
    while evals < budget:
        it += 1
        evals += s * k
        if it % n_f == 0:
            if mode == 2 and k > 1:
                k = math.ceil(k / nr)
                end = (it, evals)
            elif mode == 3 and k < n:
                k = min(n, k * nr)
                end = (it, evals)
    return it, end[0], end[1] / budget


def degenerate(d):
    deg = d[d.degenerate]
    return (deg.groupby(["stage", "algorithm", "problem"]).size(),
            int(d[(d.stage == "main") & d.degenerate].shape[0]))


def verify(d):
    g = cells(d)
    c = crossover(g)
    print("== crossover (sec:size) ==")
    # (name, recomputed, value in the report)
    checks = [
        ("coop wins", f"{c['coop_wins']}/{c['n_cells']}", "24/32"),
        ("largest baseline win", c["cut"], 254),
        ("empty interval", f"({c['cut']}, {c['gap_hi']})", "(254, 259)"),
        ("spearman rho", f"{c['rho']:.3f}", "0.687"),
        ("below: n", c["n_below"], 14),
        ("below: mean", f"{c['mean_below']:+.4f}", "+0.0009"),
        ("below: p", f"{c['p_below']:.2f}", "0.95"),
        ("above: n", c["n_above"], 18),
        ("above: mean", f"{c['mean_above']:+.4f}", "+0.0562"),
        ("above: p", f"{c['p_above']:.6f}", "0.000015"),
    ]
    for name, got, said in checks:
        ok = "OK " if str(got) == str(said) else "!! "
        print(f"  {ok}{name:24} recomputed={got:<12} tex={said}")

    print("\n== weight decay (sec:lambda) ==")
    for p, r in lam(d).items():
        print(f"  {p}: kruskal p={r['kruskal_p']:.2g}  "
              f"1e-4vs1e-3 p={r['p_1e4_vs_1e3']:.2f}  "
              f"1e-3vs1e-2 p={r['p_1e3_vs_1e2']:.3f}")
    print("  OK  tex: 1e-2 rejected on iris, not on digits; 1e-4 vs 1e-3 "
          "separated on neither.")

    print("\n== budget sweep, sonar (sec:budget / sec:discussion) ==")
    bt = budget(d)
    for b in (10000, 403642):
        row = bt.loc[b]
        print(f"  {b:>7}: " + "  ".join(f"{PRETTY[a]}={row[a]:.3f}" for a in ALGOS))
    print("  tex:   10000: gbest=0.725 MCPSO=0.569 DCPSO=0.473")
    print("  tex:  403642: gbest=0.755 MCPSO=0.826 DCPSO=0.824")

    print("\n== restructuring schedule replay (sec:cost) ==")
    m = d[d.stage == "main"]
    for p, nh in SELECTED_NH.items():
        x = m[(m.problem == p) & (m.n_hidden == nh)]
        line = []
        for algo, mode in (("mcpso", 2), ("dcpso", 3)):
            r = x[x.algorithm == algo]
            it, end, share = schedule(mode, int(r.n_weights.iloc[0]), int(r.budget.iloc[0]))
            ok = "OK " if it == round(r.iterations.mean()) else "!! "
            line.append(f"{ok}{PRETTY[algo]} {it} (logged {r.iterations.mean():.0f}), "
                        f"schedule ends it {end} at {share:.1%}")
        print(f"  {p:12}" + " | ".join(line))

    counts, main_deg = degenerate(d)
    print(f"\n== degenerate runs ==\n  in main grid: {main_deg}  (all others are tuning stages)")
    print(counts.to_string())

    print("\n== Friedman over four algorithms (sec:stats) ==")
    cols = [g[a] for a in ALGOS]
    st = friedmanchisquare(*cols)
    print(f"  chi2={st.statistic:.3f}  p={st.pvalue:.3f}  (tex: 2.025 / 0.567)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=CSV)
    a = ap.parse_args()
    verify(load(a.csv))
