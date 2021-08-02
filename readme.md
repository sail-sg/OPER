# Dependencies

- python 3.7
- pytorch 1.7.1
- hydra 1.0
- d4rl
- gym
- mujoco

# Running the code

## Data

Data must be saved into a replay buffer in the ```data/``` directory by running e.g. ```python d4rl_to_replay.py --name halfcheetah-medium-v2``` for whichever D4RL dataset you want data from.

## Training

The train loop is in ```train.py``` which can capture all algorithms by varying parameters as described below.

**Important**: to run the training files, you need to including the path from the root to the ```onestep-rl``` directory on your machine as the ```path``` variable in the corresponding config file, e.g. ```/path/to/onestep-rl```.

### Setting the training loop hyperparameters
Config files with all the relevant hyperparameters can be found in the ```config/``` directory. 

To get the one-step algorithm from the paper we set the following parameters of the training loop in ```config/train.yaml```:
```
betasteps: 5e5
steps: 1
qsteps: 2e6
pisteps: 1e5
```
To get the iterative algorithms we load the pre-trained beta and q estimators used by the one-step algorithm and then run
```
betasteps: 0
steps: 1e5
qsteps: 2
pisteps: 1
```
For the multi-step algorithms we load the pre-trained beta and q estimators used by the one-step algorithm and then run
```
betasteps: 0
steps: 5
qsteps: 2e5
pisteps: 2e4
```

## Figures

All the figures from the paper along with the notebooks that generated them are in the ```figures/``` directory. Due to space constraints, the data and log files needed to generate the figures are not included.


# Citation

If you use this repo in you research, please cite the paper as follows

```
@article{brandfonbrener2021offline,
  title={Offline RL Without Off-Policy Evaluation},
  author={Brandfonbrener, David and Whitney, William F and Ranganath, Rajesh and Bruna, Joan},
  journal={arXiv preprint arXiv:2106.08909},
  year={2021}
}
```