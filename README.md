# OPER: Offline Prioritized Eexperience Replay

This `bandit` branch performs bandit experiments to demonstrate the effectiveness of OPER.

### Usage
1. Generate an offline bandit dataset and iterative prioritized datasets by running `bandit.ipynb`.

2. Train TD3+BC on these datasets.
```
# train on the original dataset
python main_bandit.py --bc_eval=0
# train on 5th prioritized dataset by resampling
python main_bandit.py --iter=5 --resample
# train on 5th prioritized dataset by reweighting
python main_bandit.py --iter=5 --reweight
```
