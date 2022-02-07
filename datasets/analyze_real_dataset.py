import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from termcolor import colored

from datasets.utils import process
from utils.distributions import BatchOfLogit


def analyze_auto_loan():
    if os.path.exists("datasets/auto-loan/parameters.json"):
        return
    print(colored("Analyzing dataset auto loan", "blue"))

    try:
        df = pd.read_csv("datasets/auto-loan/CPRM_AutoLOan_OnlineAutoLoanData.csv")
    except:
        print(colored("Auto loan dataset has not been downloaded!", "red"))
        exit()
    rate = 0.12e-2
    df["price"] = (
        df["mp"]
        * np.power((1 + rate), -df["Term"])
        * (np.power(1 + rate, df["Term"] - 1) - 1)
        / rate
        - df["Amount_Approved"]
    )
    df = df[
        [
            "Primary_FICO",
            "Amount_Approved",
            "onemonth",
            "Competition_rate",
            "price",
            "apply",
        ]
    ]
    df = process(
        df,
        missing_value=[""],
    )
    y = df["apply"].tolist()
    x = df.drop(columns=["apply"]).to_numpy()
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(x, y)
    slope = clf.coef_[0][-1]
    df_new = df.copy()
    df_new = df_new.drop(columns=["apply"])
    df_new["price"] = 0
    x = df_new.to_numpy()
    res = clf.predict_proba(x)
    res = np.array([ele[1] for ele in res])
    biases = -np.log(1 / res - 1)
    np.random.seed(0)
    biases = np.random.choice(biases, 10000)
    res = dict()
    res["slope"] = slope
    res["biases"] = biases.tolist()
    with open("datasets/auto-loan/parameters.json", "w") as f:
        f.write(json.dumps(res))
        f.close()


def analyze_vaccine():
    if os.path.exists("datasets/vaccine/parameters.json"):
        return
    print(colored("Analyzing dataset vaccine", "blue"))

    try:
        df = pd.read_csv("datasets/vaccine/SND_0987.csv")
    except:
        print(colored("Vaccine dataset has not been downloaded!", "red"))
        exit()
    df = process(
        df,
        missing_value=[""],
        features_to_drop=[
            "SND_study",
            "SND_dataset",
            "SND_version",
            "id",
            "certainanswer",
            "choicord",
        ],
        categorical_features=[
            "age1830",
            "age3145",
            "age4665",
            "age66plus",
            "female",
            "lowincome",
            "university",
            "urban",
            "lowtberiskhome",
            "hightberiskhome",
            "summerhouserisk",
            "outTBEarea",
            "worktickrisk",
            "tickbiteever",
            "diseaseexperience",
            "healthrisktickbite",
            "lowtrustvaccinerec",
            "tbevaccinated",
            "swedishborne",
            "hightrusthealthcare",
            "outdoorpet",
        ],
    )
    y = df["buy"].tolist()
    x = df.drop(columns=["buy"]).to_numpy()
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(x, y)
    slope = clf.coef_[0][5]
    df_new = df.copy()
    df_new = df_new.drop(columns=["buy"])
    df_new["price"] = 0
    x = df_new.to_numpy()
    res = clf.predict_proba(x)
    res = np.array([ele[1] for ele in res])
    biases = -np.log(1 / res - 1)
    res = dict()
    res["slope"] = slope
    res["biases"] = biases.tolist()
    with open("datasets/vaccine/parameters.json", "w") as f:
        f.write(json.dumps(res))
        f.close()


def get_real_distribution(dataset):
    if dataset == "auto-loan":
        analyze_auto_loan()
    else:
        analyze_vaccine()
    with open(os.path.join("datasets", dataset, "parameters.json"), "r") as f:
        params = json.loads(f.read())
        f.close()
    distribution = BatchOfLogit(params["biases"], params["slope"], dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    distribution.biases = distribution.biases.to(device)
    return distribution
