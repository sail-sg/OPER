# OPER: Offline Prioritized Experience Replay

This `iql` branch performs the iql case study to demonstrate the effectiveness of OPER.

### Usage

#### Mujoco
Naive IQL that train on the original dataset:
``` 
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i --config=configs/mujoco_config.py
```

To reproduce the main results of OPER-A in the paper, i.e., only prioritizing data for policy constraint and improvement terms, train on 4 th prioritized dataset by resampling or resampling:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --weight_path=$PATH --bc_eval=true --iter 4 --std 2.0 --sampler=return-balance --two_sampler=true --config.temperature=1.0

# reweight
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --weight_path=$PATH --bc_eval=true --iter 4 --std 2.0 --reweight=true --reweight_eval=false --config.temperature=1.0
```

<!-- To prioritize data for all terms (in the ablation study), run the code:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --weight_path=$PATH --bc_eval=true --iter 4 --std 2.0 --sampler=return-balance --reweight=false 

# reweight
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --weight_path=$PATH --bc_eval=true --iter 4 --std 2.0 --reweight=true
``` -->

To reproduce the main results of OPER-R in the paper, run the code:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --sampler=return-balance

# reweight
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/mujoco_config.py --reweight=true
```

#### Antmaze
Naive IQL that train on the original dataset:
``` 
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000
```

To reproduce the main results of OPER-A in the paper:
```
# reweight
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --weight_path=$PATH --bc_eval=true --iter=3 --std=5.0 --reweight=true --reweight_eval=false
```

To reproduce the main results of OPER-R in the paper, run the code:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/antmaze_config.py --eval_episodes=100 --eval_interval=100000 --sampler=return-balance
```

#### Kitchen and Adroit
Naive IQL that train on the original dataset:
``` 
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/kitchen_config.py
```

To reproduce the main results of OPER-A in the paper:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/kitchen_config.py --weight_path=$PATH --bc_eval=true --iter 4 --std 0.5 --sampler=return-balance
```

To reproduce the main results of OPER-R in the paper, run the code:
```
# resample
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_iql.py --env_name=${env} --seed=$i  --config=configs/kitchen_config.py --bc_eval=false --sampler=return-balance
```



