import json
import os

import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from utils.distributions import Distribution

GAP_COLOR = "#256CA8"
RATIO_COLOR = "#CD2A2A"
GAP_STYLE = "-"
RATIO_STYLE = "--"
LINEWIDTH = 4


def load(dataset: str, type="gap"):
    filename = os.path.join("results", "{}_{}.json".format(dataset, type))
    if not os.path.exists(filename):
        return [], []
    with open(filename, "r") as f:
        res: dict = json.loads(f.read())
        f.close()
    revenues = [ele["revenue"] for ele in res.values()]
    consumers = [ele["consumer"] for ele in res.values()]
    return revenues, consumers


def plot_welfare(dis: Distribution, ax, fontsize: int = 20):
    revenues_gap, consumers_gap = load(dis.name, "gap")
    revenues_ratio, consumers_ratio = load(dis.name, "ratio")

    max_welfare = dis.area_all()
    min_revenue = np.min(revenues_gap)
    max_cs = max_welfare - min_revenue

    x = np.arange(0, max_cs, max_cs / 100)
    y = max_welfare - x
    ax.fill_between(x, min_revenue, y, facecolor="lightgrey", label="All feasible area")
    ax.tick_params(labelsize=int(fontsize * 0.5))
    ax.plot(
        consumers_gap,
        revenues_gap,
        label=r"$\epsilon$-difference",
        color=GAP_COLOR,
        linestyle=GAP_STYLE,
        linewidth=LINEWIDTH,
    )
    ax.plot(
        consumers_ratio,
        revenues_ratio,
        label=r"$\gamma$-ratio",
        color=RATIO_COLOR,
        linestyle=RATIO_STYLE,
        linewidth=LINEWIDTH,
    )

    crit_filename = os.path.join(
        "results", "{}_gap_critical_point.json".format(dis.name)
    )
    if os.path.exists(crit_filename):
        with open(crit_filename, "r") as f:
            res: dict = json.loads(f.read())
            f.close()
        revenues = [ele["revenue"] for ele in res.values()]
        consumers = [ele["consumer"] for ele in res.values()]
        if "power" in dis.name:
            ax.scatter(
                consumers,
                revenues,
                marker="x",
                color="black",
                label=r"$\epsilon_0$",
                linewidths=3,
                s=150,
            )

    ax.set_title(dis.plot_name(), fontsize=fontsize)


def plot_hazard(ax, dis: Distribution):
    hazards = dict()
    survival = dict()
    delta = dis.max_gap() / 50000
    for i in np.arange(0, dis.max_gap(), dis.max_gap() / 100):
        hazard = (
            (
                dis.probability_above(torch.tensor(i))
                - dis.probability_above(torch.tensor(i + delta))
            )
            / delta
            / dis.probability_above(torch.tensor(i))
        )
        hazards[i] = hazard
        survival[i] = dis.probability_above(torch.tensor(i))
    ax.plot(hazards.keys(), hazards.values())
    if "exponential" in dis.name:
        ax.set_ylim(0, 2)


def plot_all(fig, ax, dis: Distribution, fontsize):
    plot_welfare(dis, ax, fontsize=fontsize)
    inset_ax = inset_axes(ax, height="40%", width="40%", loc=1)
    plot_hazard(inset_ax, dis)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    inset_ax.set_xlabel("Hazard rate", fontsize=fontsize / 10.0 * 8)
    lines, labels = ax.get_legend_handles_labels()
    fig.legend(
        lines,
        labels,
        ncol=1,
        loc="right",
        bbox_to_anchor=(1.55, 0.5),
        labelspacing=1.2,
        fontsize=fontsize / 10.0 * 7,
    )
