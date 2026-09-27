#!/usr/bin/env python3
"""Figure for sec:discussion: validation score vs evaluation budget (Sonar).

    python3 plot_budget.py                    # writes report/fig-budget.png
    python3 plot_budget.py -o other.png
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyse import ALGOS, CSV, GBEST, PRETTY, budget, load

OUT = "report/fig-budget.png"

# (marker, line style, fill). Only the gbest baseline is filled.
STYLE = {"gbest": ("o", "-", "black"), "randgroup": ("s", "--", "white"),
         "mcpso": ("^", "-.", "white"), "dcpso": ("v", ":", "white")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--problem", default="sonar")
    a = ap.parse_args()

    t = budget(load(a.csv), a.problem)

    fig, ax = plt.subplots(figsize=(3.5, 2.25))
    for algo in ALGOS:
        marker, ls, fill = STYLE[algo]
        ax.plot(t.index, t[algo], marker=marker, linestyle=ls, color='black',
                markersize=3.5, linewidth=0.9, markerfacecolor=fill,
                markeredgewidth=0.8, label=PRETTY[algo], zorder=3)

    ax.set_xscale('log')
    ax.set_xlabel('Evaluation budget', fontsize=8)
    ax.set_ylabel('Validation macro-averaged F1', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', color='0.88', linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower right', ncol=2,
              handletextpad=0.3, borderpad=0.2, labelspacing=0.3,
              columnspacing=1.0)

    fig.tight_layout(pad=0.4)
    fig.savefig(a.out, dpi=300)
    lo, hi = t.index.min(), t.index.max()
    print(f"wrote {a.out}  ({a.problem}, {len(t)} budgets from {lo} to {hi})")
    print(f"  {GBEST} moves {t[GBEST].iloc[0]:.3f} -> {t[GBEST].iloc[-1]:.3f}; "
          f"dcpso {t['dcpso'].iloc[0]:.3f} -> {t['dcpso'].iloc[-1]:.3f}")


if __name__ == "__main__":
    main()
