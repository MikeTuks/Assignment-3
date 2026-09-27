#!/usr/bin/env python3
"""Figure for sec:size: best cooperative score minus gbest, vs network size.
One point per problem and architecture (32). Dashed line at the largest
gbest win (254 weights).
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyse import CLASSIF, CSV, TITLE, cells, crossover, load

OUT = "report/fig-crossover.png"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=OUT)
    ap.add_argument("--csv", default=CSV)
    a = ap.parse_args()

    g = cells(load(a.csv))
    cut = crossover(g)["cut"]

    # One IEEE column is ~3.5 in wide.
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    # Shade where the cooperative variants win.
    ax.axvspan(cut, 5000, color='0.93', zorder=0)

    # Marker shape = task type.
    for probs, marker, label in [(CLASSIF, 'o', 'Classification'),
                                 (None, '^', 'Approximation')]:
        sub = g[g.problem.isin(probs)] if probs else g[~g.problem.isin(CLASSIF)]
        ax.scatter(sub.n_weights, sub.margin, s=20, marker=marker,
                   color='black', linewidths=0.8, label=label, zorder=3)

    ax.axhline(0, color='0.5', linewidth=0.7, zorder=1)
    ax.axvline(cut, color='black', linestyle='--', linewidth=0.8, zorder=2)

    ax.set_xscale('log')
    ax.set_xlim(9, 5000)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.06)          # headroom for the region labels

    # Y-axis direction label, in the empty upper-left corner.
    ax.annotate('cooperative\nbetter', xy=(0.075, 0.96), xycoords='axes fraction',
                fontsize=7.5, style='italic', ha='left', va='top', color='0.25',
                linespacing=1.25)
    ax.annotate('', xy=(0.045, 0.96), xytext=(0.045, 0.72),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', color='0.45', linewidth=0.7,
                                shrinkA=0, shrinkB=0))
    ax.annotate(f'{cut} weights', xy=(cut, hi + 0.045), xytext=(4, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=7.5)

    ax.set_xlabel('Network size (number of weights)', fontsize=8)
    ax.set_ylabel('Score difference', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', color='0.88', linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc='lower right',
              handletextpad=0.3, borderpad=0.2, labelspacing=0.3)

    fig.tight_layout(pad=0.4)
    fig.savefig(a.out, dpi=300)
    print(f"wrote {a.out}  ({len(g)} points, cut at {cut})")


if __name__ == "__main__":
    main()
