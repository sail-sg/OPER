from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent: nn.Module, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}

    for _ in range(num_episodes):
        observation, done = env.reset(), False

        while not done:
            action = agent.sample_actions(observation, temperature=0.0) # determinisitc when T=0
            observation, _, done, info = env.step(action)

        for k in stats.keys():
            stats[k].append(info['episode'][k])

    returns = stats['return']
    print(f'mean is {np.mean(returns).round(1)}, std is {np.std(returns).round(1)}')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats
