# OPER: Offline Prioritized Experience Replay

OPER is a plug-and-play component for offline RL algorithms, where comprises two stage: (1) calculate static priority weights. (2) Train offline RL algorithms on the prioritized dataset by resampling or reweighting.
The paper can be found [here](). 

As case studies, we evaluate OPER on five different algorithms, including BC, TD3+BC, Onestep RL, CQL, and IQL on D4RL. Extensive experiments demonstrate that both OPER-A and OPER-R significantly improve the performance for all baseline methods.
We make public the priority weights of D4RL benchmark [here](). 

### Usage

#### OPER-A
To calculate (unnormalized) OPER-A weights, run the code in `main` branch by:
```
python main.py --env hopper-medium-expert-v2 --seed 1 --n_step 1  --iter 5  --first_eval_steps 1000000 --bc_eval_steps 1000000 
```
Except for kitchen and pen:
```
python main.py --env kitchen-complete-v0 --seed 1 --n_step 5  --iter 5  --first_eval_steps 500000 --bc_eval_steps 500000 
```
In the paper, we run 3 seeds and get the average to reduce variance of OPER-A weights. 

#### OPER-R
The code of getting (unnormalized) OPER-R weights is [here](https://github.com/yueyang130/TD3_BC/blob/9285f1c0ce95cc5e2b8c4eb52fccccb6c7b523bd/utils.py#L174), which is also included in the case study codes. No need for extra running to get OPER-R weights.

#### Visualization 
Visualization of OPER-A and OPER-R weights can be found in `load_weights.ipynb`. 

#### Case Studies
We provide the codes for cases studies (bandit, BC, TD3+BC, IQL, CQL, OnestepRL), which can be found in the corresponding branches. The code for BC is at the `OnestepRL` branch. Detailed usages is at `README.md` in these branches. 
Note that for re-sampling and re-weighting, the priority weights from the first stage should be normalized. The normalization has been included in the codes of the second stage, no need for extra processing.

### Bibtex
```

```

### *Note
The code in this repository has been organized. Errors that may arise during the organizing process could lead to code malfunctions or discrepancies from the original research results. If you encounter any problems, please raise issues. I will go and fix these bugs.


---

