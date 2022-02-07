import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from termcolor import colored
from torch import nn
from tqdm import tqdm

from utils.distributions import Distribution


class PricingGap(nn.Module):
    def __init__(self, dis: Distribution, epsilon: float):
        super(PricingGap, self).__init__()
        self.dis = dis
        self.epsilon = epsilon
        self.price = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float64) * dis.initialize(), requires_grad=True
        )

    def revenue(self):
        price_upper = self.price + self.epsilon
        r = price_upper * self.dis.probability_above(price_upper)
        r += self.dis.area_between(self.price, price_upper)
        return r

    def consumer_surplus(self):
        price_upper = self.price + self.epsilon
        s = self.dis.area_above(price_upper)
        s -= price_upper * self.dis.probability_above(price_upper)
        return s

    def result(self):
        return [self.price.item(), (self.price + self.epsilon).item()]


def get_welfare_gap(
    dis: Distribution, epsilon: float, num_iter: int = 300000, lr: float = 5e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricing = PricingGap(dis, epsilon).to(device)
    optim = torch.optim.SGD(pricing.parameters(), lr=lr)
    max_revenue = -10000
    max_i = -1
    for i in tqdm(range(num_iter)):
        revenue = -pricing.revenue()
        optim.zero_grad()
        revenue.backward()
        optim.step()

        now_revenue = -revenue.item()
        if i <= 1000:
            continue
        if now_revenue > max_revenue:
            max_revenue = now_revenue
            max_i = i
        tolerence = 20
        if i - max_i >= tolerence:
            break

    return pricing.result(), pricing.revenue().item(), pricing.consumer_surplus().item()


def calc_welfare_gap(dis: Distribution, lr: float = 5e-3):
    if not os.path.exists("results"):
        os.mkdir("results")
    filename = os.path.join("results", "{}_gap.json".format(dis.name))
    if os.path.exists(filename):
        return

    print(
        colored(
            "Analyzing the welfare under gap constraint for distribution {}".format(
                dis.name
            ),
            "blue",
        )
    )

    res = dict()
    cands = list(np.arange(0, dis.max_gap(), dis.max_gap() / 50))
    if not "uniform" in dis.name:
        cands.append(dis.max_gap().item() * 10)
    for i, epsilon in enumerate(cands):
        p, r, cs = get_welfare_gap(dis, epsilon, lr=lr)
        print(
            "Distribution {}, gap {:.3f}, pricing [{:.3f}, {:.3f}] revenue {:.5f}, consumer surplus {:.5f}".format(
                dis.name, epsilon, p[0], p[1], r, cs
            )
        )
        res[epsilon] = {}
        res[epsilon]["price"] = p
        res[epsilon]["revenue"] = r
        res[epsilon]["consumer"] = cs

    with open(filename, "w") as f:
        f.write(json.dumps(res, indent=4))
        f.close()


def analyze_critical_point(dis: Distribution, lr: float = 5e-3):
    filename = os.path.join("results", "{}_gap.json".format(dis.name))
    crit_filename = os.path.join(
        "results", "{}_gap_critical_point.json".format(dis.name)
    )
    if not os.path.exists(filename) or os.path.exists(crit_filename):
        return

    print(
        colored(
            "Analyzing the critical point for distribution {}".format(dis.name), "blue"
        )
    )

    with open(filename, "r") as f:
        res: dict = json.loads(f.read())
        f.close()

    out_res = dict()
    epsilon_critical = []
    for k, v in res.items():
        eps = float(k)
        p_l = float(v["price"][0])
        if eps - 2 * p_l >= 0:
            epsilon_critical.append(eps)
            break

    for crit in epsilon_critical:
        p, r, cs = get_welfare_gap(dis, crit, lr=lr)
        out_res[crit] = {}
        out_res[crit]["price"] = p
        out_res[crit]["revenue"] = r
        out_res[crit]["consumer"] = cs

        print(
            "{} Critical point, pricing [{:.3f}, {:.3f}] revenue {:.5f}, consumer surplus {:.5f}".format(
                dis.name, p[0], p[1], r, cs
            )
        )

    with open(crit_filename, "w") as f:
        f.write(json.dumps(out_res, indent=4))
        f.close()
