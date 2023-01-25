import numpy as np
import torch
import gym
import argparse
import os
import d4rl
import wandb
import utils
import TD3_BC
import time
from advantage import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# python main_bandit.py --resample
# python main_bandit.py

def vis(actions, colors='grey', path=None, title=None, legend=None):
    scatter = plt.scatter(actions[:, 0], actions[:, 1],
         c=colors, alpha=0.1)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    if title: plt.title(title, fontsize = 25, fontweight="normal")
    if path: plt.savefig(path, bbox_inches='tight')


def eval_policy(policy, steps, eval_episodes=100, mini_batch=1000):
    mini_batch = min(mini_batch, eval_episodes)
    actions = []
    for _ in range(eval_episodes//mini_batch):
        if args.random_state:
            state = np.random.randn(mini_batch,1)
        else:
            state = np.zeros((mini_batch,1), dtype=np.float32)
        action = policy.batch_select_action(state)
        actions.append(action)
    actions = np.concatenate(actions, axis=0)
    if args.bc_eval:
        title=f'{args.iter}th prioritized dataset'
    else:
        title=f'the original dataset'
    vis(actions, path=f'{path}/step{steps+1}.pdf', title=title)
    np.save(f'{path}/step{steps+1}.npy', actions)

def none_or_float(value):
    if value == 'None' or value == 'none':
        return None
    return float(value)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # Experiment
    parser.add_argument("--policy", default="TD3_BC")               # Policy name
    parser.add_argument("--env", default="bandit")        # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--log_freq", default=1e3, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_freq", default=1e4, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--eval_episodes", default=100, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e5, type=int)   # Max time steps to run environment
    parser.add_argument("--random_state", default=0, type=int)  
    parser.add_argument("--optimistic_init", default=1, type=int)   # Optimistically initialize Q-value. Equivalently, decrease rewards.
    # load weight
    parser.add_argument("--bc_eval", type=int, default=1)   
    parser.add_argument("--iter", type=int, default=5, help='K th rebalanced behavior policy.')       
    parser.add_argument("--weight_func", default='linear', choices=['linear', 'exp', 'power'])    
    parser.add_argument("--std", default=None, type=none_or_float, help="scale weights' standard deviation.")    
    parser.add_argument("--eps", default=None, type=none_or_float, help="")    
    parser.add_argument("--eps_max", default=None, type=none_or_float, help="")    
    # TD3
    parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--tau", default=0.005)                     # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    # TD3 + BC
    parser.add_argument("--alpha", default=0.25, type=float)
    parser.add_argument("--normalize", default=True)
    # rebalance
    parser.add_argument("--base_prob", default=0.0, type=float)
    parser.add_argument("--resample", action="store_true")
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--reweight_eval", default=1, type=int)
    parser.add_argument("--reweight_improve", default=1, type=int)
    parser.add_argument("--reweight_constraint", default=1, type=int)
    parser.add_argument("--clip_constraint", default=0, type=int)  # 0: no clip; 1: hard clip; 2 soft clip
    parser.add_argument("--tag", default='', type=str)
    args = parser.parse_args()

    # resample and reweight can not been applied together
    assert not args.resample or not args.reweight

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    itr = args.iter if args.bc_eval else 0 
    path = f'./results/td3_bc_bandit/iter{itr}_alpha{args.alpha}_randomstate{args.random_state}'
    if not os.path.exists(path):
        os.makedirs(path)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    state_dim = 1 # placeholder
    action_dim = 2
    max_action = 1.0

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "tau": args.tau,
        # generate weight
        "iter": args.iter,
        "bc_eval": args.bc_eval,
        "weight_func": args.weight_func,
        "std": args.std,
        "eps": args.eps,
        "eps_max": args.eps_max,
        # TD3
        "policy_noise": args.policy_noise * max_action,
        "noise_clip": args.noise_clip * max_action,
        "policy_freq": args.policy_freq,
        # TD3 + BC
        "alpha": args.alpha, 
        "reweight_eval": args.reweight_eval, 
        "reweight_improve": args.reweight_improve,
        "reweight_constraint": args.reweight_constraint,
        "clip_constraint": args.clip_constraint,
    }

    wandb.init(project="TD3_BC", config={
            "env": args.env, "seed": args.seed, "tag": args.tag,
            "resample": args.resample, "reweight": args.reweight, "p_base": args.base_prob,
            **kwargs
            })

    # replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.batch_size,
    #     base_prob=args.base_prob, resample=args.resample, reweight=args.reweight, discount=args.discount)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim, args.batch_size,
        base_prob=args.base_prob, resample=args.resample, reweight=args.reweight, n_step=1, discount=1.0)
    dataset = np.load('results/bandit.npy', allow_pickle=True).item()
    replay_buffer.convert_bandit(dataset, args.random_state, args.optimistic_init)
    

    if args.bc_eval:
        # weight loading module (filename changed)
            
        eval_res = np.load('results/weight.npy', allow_pickle=True).item()
        num_iter = eval_res['iter']
        assert args.iter <= num_iter
        weight = eval_res[f'weight{args.iter}']
        print(f'Loading weights at {args.iter}th rebalanced behavior policy')
        replay_buffer.replace_weights(weight, args.weight_func, 0, args.std, args.eps, args.eps_max)

    # Initialize policy
    policy = TD3_BC.TD3_BC(**kwargs)

    
    # time0 = time.time()
    evaluations = []
    for t in range(int(args.max_timesteps)):
        infos = policy.train(replay_buffer)
        if (t + 1) % args.log_freq == 0:
            for k, v in infos.items():
                wandb.log({f'train/{k}': v}, step=t+1)
        # Evaluate episode
        if (t + 1) % args.eval_freq == 0:
            print(f"Time steps: {t+1}")
            evaluations.append(eval_policy(policy, t))
            # wandb.log({f'eval/score': evaluations[-1]}, step=t+1)
            # wandb.log({f'eval/avg10_score': np.mean(evaluations[-min(10, len(evaluations)):])}, step=t+1)

        
