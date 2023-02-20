# OPER: Offline Prioritized Eexperience Replay

OPER is a plug-and-play component for offline RL algorithms, where comprises two stage: (1) calculate static priority weights. (2) Train offline RL algorithms on the prioritized dataset by resampling or reweighting.
The paper can be found [here]().


As case studies, we evaluate OPER on five different algorithms, including BC, TD3+BC, Onestep RL, CQL, and IQL. Extensive experiments demonstrate that both OPER-A and OPER-R signifi- cantly improve the performance for all baseline methods, achieving new state-of-the-art on the D4RL benchmark.

### Usage
To calculate (unnormalized) OPER-A weights, run the code in `main` branch by:
```
python main.py --env hopper-medium-expert --seed 1 --n_step 1  --iter 1  --first_eval_steps 1000000 --bc_eval_steps 1000000 
```
Except for kitchen and pen:
```
python main.py --env kitchen-complete-v0 --seed 1 --n_step 5  --iter 5  --first_eval_steps 500000 --bc_eval_steps 500000 
```
In the paper, we run 3 seeds and get the average to reduce vairnace of OPER-A weights. The code of getting (unnormalized) OPER-R weights is [here](https://github.com/yueyang130/TD3_BC/blob/9285f1c0ce95cc5e2b8c4eb52fccccb6c7b523bd/utils.py#L174). Visualization of OPER-A and OPER-R weights can be found in `load_weights.ipynb`. We make public the priority weights of D4RL benchmark [here](). 

Note that for re-sampling, the priority weights should be normazied by 
```
# non-negative
weight = weight - weight.min()
# sum is 1
prob = weight / weight.sum()
```
For re-weighting, the priority weights should be normazied by 
```
# non-negative
weight = weight - weight.min()
# mean is 1
prob = weight / weight.sum() * weight.size()
```

The normalization process is included in the codes for cases studies (TD3+BC, IQL, CQL, OnestepRL), which can be found in other branches.

### Bibtex
```

```

---

