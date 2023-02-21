# OPER: Offline Prioritized Eexperience Replay

This `onestep` branch performs the BC and OnestepRL case study to demonstrate the effectiveness of OPER. For simplicity, we only provide re-sampling implementation for OPER in this branch.

## Dependencies

Requirements can be installed via pip by running:
```
pip install -r requirements.txt
```


## Running the code

Offline dataset must be saved into a replay buffer in the ```data/``` directory by running e.g. ```python d4rl_to_replay.py --name halfcheetah-medium-v2``` for whichever D4RL dataset you want data from.

**Important**: to run the training files, you need to include the path from the root to the ```onestep-rl``` directory on your machine as the ```path``` variable in the ```config/train.yaml``` file, e.g. ```path: /path/to/onestep-rl```.

To get the one-step algorithm from the paper we set the following parameters of the training loop in ```config/train.yaml```:
```
beta_steps: 5e5
steps: 1
qs_teps: 2e6
pi_steps: 1e5
```

## Onestep Case Study Usage

Train OnestepRL on Gym locomotion tasks.

Train on the original dataset
``` 
python train.py env.name=$env seed=$i pi.temp=1.0
```

To reproduce the main results of OPER-A in the paper, i.e., only prioritizing data for policy constraint and improvement terms, train on 4 th prioritized dataset by resampling:
```
python train.py env.name=$env seed=$i pi.temp=1.0 resampling=adv two_sampler=true weight_path=$PATH iter=4 std=2.0
```

To prioritize data for all terms, run the code:
```
python train.py env.name=$env seed=$i pi.temp=1.0 resampling=adv weight_path=$PATH iter=4 std=2.0
```

To reproduce the main results of OPER-R in the paper, run the code:
```
python train.py env.name=$env seed=$i pi.temp=1.0 resampling=return
```

## BC Case Study Usage

Train on the original dataset
``` 
python train.py env.name=$env seed=$i train_q=False train_pi=False 
```

To reproduce the main results of OPER-A in the paper, train on 4 th prioritized dataset by resampling:
```
python train.py env.name=$env seed=$i train_q=False train_pi=False resampling=adv weight_path=$PATH iter=4 std=2.0
```

To reproduce the main results of OPER-R in the paper, run the code:
```
python train.py env.name=$env seed=$i train_q=False train_pi=False resampling=return
```
