from typing import Callable, Sequence, Tuple, Optional

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        if self.encoder:
            observations = self.encoder(observations)
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Optional[nn.Module] = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        if self.encoder:
            observations = self.encoder(observations)
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Optional[nn.Module] = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self.encoder:
            observations = self.encoder(observations)
        critic1 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
