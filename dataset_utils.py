import collections
from typing import Optional
from typing import Any
import d4rl
import gym
import numpy as np
from tqdm import tqdm

Array = Any


Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class RandSampler(object):
  """A random sampler."""

  def __init__(self, max_size: int, batch_size: int = 1) -> None:
    self._max_size = max_size
    self._batch_size = batch_size

  def sample(self):
    """Return an array of sampled indices."""
    return np.random.randint(self._max_size, size=self._batch_size)


class PrefetchBalancedSampler(object):
  """A prefetch balanced sampler."""

  def __init__(self, probs: Array, max_size: int, batch_size: int, n_prefetch: int) -> None:
    self._max_size = max_size
    self._batch_size = batch_size
    self.n_prefetch = min(n_prefetch, max_size//batch_size)
    self._probs = probs / np.sum(probs)
    self.cnt = self.n_prefetch - 1
  
  def sample(self):
    self.cnt = (self.cnt+1)%self.n_prefetch
    if self.cnt == 0:
      self.indices = np.random.choice(self._max_size, 
          size=self._batch_size * self.n_prefetch, p=self._probs)
    return self.indices[self.cnt*self._batch_size : (self.cnt+1)*self._batch_size]


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int, batch_size: int, sample: str, base_prob: float):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size
        self.returns = self.compute_return()
        self.batch_size = batch_size
        self.sampler = self.make_sampler(sample, base_prob)

    def compute_return(self):
        returns, ret, start = [], 0, 0
        for i in range(self.size):
            ret = ret+self.rewards[i]
            if self.dones_float[i]: 
                returns.extend([ret]*(i-start+1))
                start = i + 1
                ret = 0
        assert len(returns) == self.size
        return np.stack(returns)

    def make_sampler(self, sample: str, base_prob: float):
        if sample == 'uniform':
            return RandSampler(self.size, self.batch_size)
        if 'balance' in sample:
            if 'reward' in sample:
                dist = self.rewards
            elif 'return' in sample:
                dist = self.returns
            else:
                raise NotImplemented
            if 'inverse' not in sample:
                probs = (dist - dist.min()) / (dist.max() - dist.min()) + base_prob
            else:
                probs = 1 - (dist - dist.min()) / (dist.max() - dist.min()) + base_prob
            # probs = np.sqrt(probs)
            return PrefetchBalancedSampler(probs, self.size, self.batch_size, n_prefetch=1000)
        else:
            raise NotImplemented

    def sample(self) -> Batch:
        # indx = np.random.randint(self.size, size=self.batch_size)
        indx = self.sampler.sample()

        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 batch_size: int,
                 sample: str,
                 base_prob: float,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5):
        dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1

        super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']),
                         batch_size=batch_size,
                         sample=sample,
                         base_prob=base_prob)


class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
