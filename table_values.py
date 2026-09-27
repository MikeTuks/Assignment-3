#!/usr/bin/env python3
"""Prints the values for each table in the report (plain text, no LaTeX).

    python3 table_values.py                   # every table
    python3 table_values.py tab:class         # one, or several
    python3 table_values.py --csv-out         # comma-separated, for pasting
    python3 table_values.py --list            # what is available

Rounding: scores 4 dp, recall 3 dp, iterations whole numbers.
"""
import argparse
import sys

import pandas as pd

from analyse import (ALGOS, APPROX, CLASSIF, CSV, PRETTY, SELECTED_NH, TITLE,
                     load, selected)

HEAD = ["Problem", "nh"] + [PRETTY[a] for a in ALGOS]


def _scores(sel, problems, dp=4):
    rows = []
    for p in problems:
        row = [TITLE[p], str(SELECTED_NH[p])]
        for a in ALGOS:
            row.append(f"{sel[('mean', a)][p]:.{dp}f} +/- {sel[('std', a)][p]:.3f}")
        rows.append(row)
    return rows


def tab_selected(d):
    return HEAD, _scores(selected(d), CLASSIF + APPROX)


def tab_class(d):
    return HEAD, _scores(selected(d), CLASSIF)


def tab_approx(d):
    return HEAD, _scores(selected(d), APPROX)


def tab_recall(d):
    m = d[d.stage == "main"]
    dig = m[(m.problem == "digits") & (m.n_hidden == SELECTED_NH["digits"])]
    r = dig.groupby("algorithm")["test_min_recall"].agg(["mean", "std"])
    return (["Measure"] + [PRETTY[a] for a in ALGOS],
            [["Minimum class recall"] +
             [f"{r['mean'][a]:.3f} +/- {r['std'][a]:.3f}" for a in ALGOS]])


def tab_iters(d):
    m = d[d.stage == "main"]
    probs = ["iris", "sonar", "digits", "communities"]
    it = (m.merge(pd.Series(SELECTED_NH, name="nh"), left_on="problem",
                  right_index=True)
           .query("n_hidden == nh and problem in @probs")
           .groupby(["problem", "algorithm"])["iterations"].mean().unstack())
    return HEAD, [[TITLE[p], str(SELECTED_NH[p])] +
                  [f"{it[a][p]:,.0f}" for a in ALGOS] for p in probs]


def tab_noise(d):
    m = d[d.stage == "main"]
    nz = (m[m.problem.isin(["concrete", "friedman"]) & (m.n_hidden == 10)]
          .groupby(["problem", "algorithm"])["test_score"].agg(["mean", "std"]))
    return (["Problem"] + [PRETTY[a] for a in ALGOS],
            [[TITLE[p]] + [f"{nz['mean'][(p, a)]:.4f} +/- {nz['std'][(p, a)]:.3f}"
                           for a in ALGOS]
             for p in ["concrete", "friedman"]])


def tab_nhmodes(d):
    h = d[d.stage == "hidden"]
    pick = (h.groupby(["budget_mode", "problem", "n_hidden"])["val_score"].mean()
             .reset_index().sort_values("val_score")
             .groupby(["budget_mode", "problem"]).tail(1))
    modes = sorted(pick.budget_mode.unique())
    rows = []
    for p in CLASSIF + APPROX:
        row = [TITLE[p]]
        for md in modes:
            s = pick[(pick.budget_mode == md) & (pick.problem == p)]
            row.append(str(int(s.n_hidden.iloc[0])) if len(s) else "--")
        rows.append(row)
    return ["Problem"] + [m.capitalize() for m in modes], rows


TABLES = {
    "tab:nhmodes": ("Architecture selected under each budget mode", tab_nhmodes),
    "tab:selected": ("Test score at the selected architecture", tab_selected),
    "tab:class": ("Macro-F1, classification problems", tab_class),
    "tab:recall": ("Recall of the worst-recovered digit", tab_recall),
    "tab:approx": ("Negative RMSE, function approximation", tab_approx),
    "tab:iters": ("Iterations afforded within the budget", tab_iters),
    "tab:noise": ("Response to uninformative inputs", tab_noise),
}


def show(label, head, rows, as_csv):
    title = TABLES[label][0]
    if as_csv:
        print(f"# {label} -- {title}")
        print(",".join(head))
        for r in rows:
            print(",".join(r))
        print()
        return
    w = [max(len(head[i]), *(len(r[i]) for r in rows)) for i in range(len(head))]
    bar = "  ".join("-" * x for x in w)
    print(f"{label} -- {title}")
    print("  " + "  ".join(h.ljust(w[i]) for i, h in enumerate(head)))
    print("  " + bar)
    for r in rows:
        print("  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", nargs="*", help="e.g. tab:class (default: all)")
    ap.add_argument("--csv-out", action="store_true", help="comma-separated output")
    ap.add_argument("--list", action="store_true", help="list available tables")
    ap.add_argument("--csv", default=CSV, help="results file to read")
    a = ap.parse_args()

    if a.list:
        for k, (desc, _) in TABLES.items():
            print(f"  {k:16} {desc}")
        return

    wanted = a.labels or list(TABLES)
    unknown = [w for w in wanted if w not in TABLES]
    if unknown:
        sys.exit(f"unknown: {unknown}\nknown: {list(TABLES)}")

    d = load(a.csv)
    for label in wanted:
        head, rows = TABLES[label][1](d)
        show(label, head, rows, a.csv_out)


if __name__ == "__main__":
    main()
