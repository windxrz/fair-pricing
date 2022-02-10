# Fair pricing
Source code for WWW 2022 paper [Regulatory Instruments for Fair Personalized Pricing](https://arxiv.org/abs/2202.04245).

## Installation
### Requirements
- Linux with Python >= 3.6
- `pip install -r requirements.txt`

## Quick Start
### Plot the trade-off curve between consumer surplus and producer surplus for common demand distributions
```bash
python main.py --distribution DISTRIBUTION
```

Here DISTRIBUTION takes value from `uniform`, `exponential`, `power-law`, `coke`, `cake`, `vaccine`, or `auto-loan`. The figures will be shown in `results/figs/`.

## Citing
If you find this repo useful for your research, please consider citing the paper.
```
@article{xu2022regulatory,
  title={Regulatory Instruments for Fair Personalized Pricing},
  author={Xu, Renzhe and Zhang, Xingxuan and Cui, Peng and Li, Bo and Shen, Zheyan and Xu, Jiazheng},
  journal={arXiv preprint arXiv:2202.04245},
  year={2022}
}
