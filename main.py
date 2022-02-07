import argparse
import json
import os

import matplotlib
import torch
from matplotlib import pyplot as plt
from termcolor import colored

from utils.analyze_welfare_gap import analyze_critical_point, calc_welfare_gap
from utils.analyze_welfare_ratio import calc_welfare_ratio
from utils.distributions import BatchOfLogit, Exponential, Logit, PowerLaw, Uniform
from utils.plot import plot_all

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42


def get_real_distribution(dataset):
    with open(os.path.join("datasets", dataset, "parameters.json"), "r") as f:
        params = json.loads(f.read())
        f.close()
    distribution = BatchOfLogit(params["biases"], params["slope"], dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution.biases = distribution.biases.to(device)
    return distribution


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distribution",
        choices=[
            "uniform",
            "exponential",
            "power-law",
            "coke",
            "cake",
            "vaccine",
            "auto-loan",
        ],
        default="uniform",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not os.path.exists("results/figs"):
        os.makedirs("results/figs")

    if args.distribution == "uniform":
        dis = Uniform(0, 1)
    elif args.distribution == "exponential":
        dis = Exponential(1)
    elif args.distribution == "power-law":
        dis = PowerLaw(2)
    elif args.distribution == "coke":
        dis = Logit(3.94, -3.44, name="coke")
    elif args.distribution == "cake":
        dis = Logit(4.58, -3.72, name="cake")
    elif args.distribution == "vaccine":
        dis = get_real_distribution("vaccine")
    elif args.distribution == "auto-loan":
        dis = get_real_distribution("auto-loan")

    calc_welfare_gap(dis)
    calc_welfare_ratio(dis)
    if "power_law" in dis.name:
        analyze_critical_point(dis)

    plt.clf()
    fig, ax = plt.subplots(figsize=(4, 4))
    plot_all(fig, ax, dis, fontsize=20)
    print(
        colored(
            "Saving welfare trade-off curve to results/figs/{}.png".format(dis.name),
            "blue",
        )
    )
    plt.savefig("results/figs/{}.png".format(dis.name), bbox_inches="tight")


if __name__ == "__main__":
    main()
