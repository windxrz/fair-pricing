from abc import ABC, abstractmethod

import numpy as np
import torch
from matplotlib import pyplot as plt


class Distribution(ABC):
    def __init__(self, name: str):
        self.name = name

    def probability_between(self, lower, upper):
        return self.probability_above(lower) - self.probability_above(upper)

    def area_between(self, lower, upper):
        return self.area_above(lower) - self.area_above(upper)

    @abstractmethod
    def probability_above(self, lower):
        pass

    @abstractmethod
    def area_above(self, lower):
        pass

    @abstractmethod
    def area_all(self):
        pass

    @abstractmethod
    def max_gap(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def plot_name(self) -> str:
        pass


class Uniform(Distribution):
    def __init__(self, a, b):
        super().__init__("uniform_{}_{}".format(a, b))
        self.a = a
        self.b = b

    def probability_above(self, lower):
        return torch.relu((self.b - lower) / (self.b - self.a))

    def area_above(self, lower):
        return torch.relu((self.b * self.b - lower * lower) / (self.b - self.a) / 2)

    def area_all(self):
        return self.area_above(torch.tensor(self.a))

    def max_gap(self):
        return self.b - self.a

    def initialize(self):
        return self.a + (self.b - self.a) / 10

    def plot_name(self):
        return "Uniform"


class Exponential(Distribution):
    def __init__(self, lmbd):
        super().__init__("exponential_{}".format(lmbd))
        self.lmbd = lmbd

    def probability_above(self, lower):
        return torch.exp(-self.lmbd * lower)

    def area_above(self, lower):
        return (lower + 1 / self.lmbd) * torch.exp(-self.lmbd * lower)

    def area_all(self):
        return self.area_above(torch.tensor(0))

    def max_gap(self):
        return np.log(100) / self.lmbd

    def initialize(self):
        return 1 / self.lmbd

    def plot_name(self):
        return "Exponential"


class PowerLaw(Distribution):
    def __init__(self, alpha, delta=1):
        super().__init__("power_law_{}_{}".format(alpha, delta))
        self.alpha = alpha
        self.delta = delta

    def probability_above(self, lower):
        lower = torch.max(lower, torch.tensor(0)) + self.delta
        return torch.exp(
            self.alpha * (torch.log(torch.tensor(self.delta)) - torch.log(lower))
        )

    def area_above(self, lower):
        lower = torch.max(lower, torch.tensor(0))
        return (
            1
            / (self.alpha - 1)
            * torch.exp(self.alpha * torch.log(torch.tensor(self.delta)))
            * torch.exp(-self.alpha * torch.log(lower + self.delta))
            * (self.alpha * lower + self.delta)
        )

    def area_all(self):
        return self.area_above(torch.tensor(0))

    def max_gap(self):
        return np.log(1000) / self.alpha

    def initialize(self):
        return self.delta * 1.1

    def plot_name(self):
        return "Power law"


class Logit(Distribution):
    def __init__(self, a, b, name=None):
        if name is None:
            super().__init__("logit_{}_{}".format(a, b))
        else:
            super().__init__(name)
        self.a = a
        self.b = b

    def probability_above(self, lower):
        lower = torch.max(lower, torch.tensor(0))
        return 1 / (1 + torch.exp(-self.a - self.b * lower))

    def area_above(self, lower):
        lower = torch.max(lower, torch.tensor(0))
        return -torch.log(
            torch.exp(self.a + self.b * lower) + 1
        ) / self.b + lower * self.probability_above(lower)

    def area_all(self):
        return self.area_above(torch.tensor(0))

    def max_gap(self):
        return (-np.log(99) - self.a) / self.b

    def initialize(self):
        return -self.a / self.b

    def plot_name(self):
        if not "_" in self.name:
            return self.name.capitalize()
        else:
            return "Logit a = {}, b = {}".format(self.a, self.b)


class BatchOfLogit(Distribution):
    def __init__(self, biases, slope, dataset):
        super().__init__(dataset)
        self.biases = torch.tensor(biases, requires_grad=False)
        self.slope = slope
        self.dataset = dataset

    def probability_above(self, lower):
        lower = torch.max(lower, torch.tensor(0))
        res = torch.mean(
            1 / (1 + torch.exp(-self.biases - self.slope * lower)), 0
        ).view(-1)
        return res

    def area_above(self, lower):
        lower = torch.max(lower, torch.tensor(0))
        res = torch.mean(
            -torch.log(torch.exp(self.biases + self.slope * lower) + 1) / self.slope
            + lower / (1 + torch.exp(-self.biases - self.slope * lower))
        ).view(-1)
        return res

    def area_all(self):
        return self.area_above(torch.tensor(0))

    def max_gap(self):
        if "auto" in self.dataset:
            res = 15000
        else:
            res = 2000
        return res

    def initialize(self):
        res = self.max_gap() / 10
        return res

    def plot_name(self):
        return self.name
