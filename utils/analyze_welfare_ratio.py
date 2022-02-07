import json
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from termcolor import colored
from torch import nn
from tqdm import tqdm

from utils.distributions import Distribution, Uniform


class PricingRatio(nn.Module):
    def __init__(self, dis: Distribution, alpha: float):
        super(PricingRatio, self).__init__()
        self.dis = dis
        self.alpha = alpha
        self.price = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float64) * dis.initialize() / alpha,
            requires_grad=True,
        )

    def revenue(self):
        price_upper = self.price * self.alpha
        r = price_upper * self.dis.probability_above(price_upper)
        r += self.dis.area_between(self.price, price_upper)
        return r

    def consumer_surplus(self):
        price_upper = self.price * self.alpha
        s = self.dis.area_above(price_upper)
        s -= price_upper.item() * self.dis.probability_above(price_upper)
        return s

    def result(self):
        lower = self.price.item()
        upper = lower * self.alpha
        if "uniform" in self.dis.name:
            dis: Uniform = self.dis
            if upper > dis.b:
                upper = dis.b
        return [lower, upper]


def get_welfare_ratio(
    dis: Distribution, epsilon: float, num_iter: int = 300000, lr: float = 5e-3
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pricing = PricingRatio(dis, epsilon).to(device)
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


def calc_welfare_ratio(dis: Distribution, lr: float = 5e-3):
    if not os.path.exists("results"):
        os.mkdir("results")
    filename = os.path.join("results", "{}_ratio.json".format(dis.name))
    if os.path.exists(filename):
        return

    print(
        colored(
            "Analyzing the welfare under ratio constraint for distribution {}".format(
                dis.name
            ),
            "blue",
        )
    )

    res = dict()
    cands = list(np.arange(0, 5.5, 1e-1))
    if "power_law" in dis.name:
        cands.append(7)
    for i, alpha in enumerate(cands):
        alpha = np.exp(alpha)
        p, r, cs = get_welfare_ratio(dis, alpha, lr=lr)
        print(
            "Distribution {}, ratio {:.3f}, pricing [{:.3f}, {:.3f}] revenue {:.5f}, consumer surplus {:.5f}".format(
                dis.name, alpha, p[0], p[1], r, cs
            )
        )
        res[alpha] = {}
        res[alpha]["price"] = p
        res[alpha]["revenue"] = r
        res[alpha]["consumer"] = cs

    with open(filename, "w") as f:
        f.write(json.dumps(res, indent=4))
        f.close()
