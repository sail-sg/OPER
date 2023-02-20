from random import triangular
from typing import Callable, Sequence, Tuple, Optional, Any

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Any = None

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=True) -> jnp.ndarray:
        if self.encoder:
            observations = self.encoder()(observations, training)
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Any = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray, training=True) -> jnp.ndarray:
        if self.encoder:
            inputs = jnp.concatenate([observations, actions], -1)
            inputs = self.encoder()(inputs, training)
        else:
            inputs = jnp.concatenate([observations, actions], -1)

        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    encoder: Any = None
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray, training=True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        critic1 = Critic(self.hidden_dims, encoder=self.encoder,
                         activations=self.activations)(observations, actions)
        critic2 = Critic(self.hidden_dims, encoder=self.encoder,
                         activations=self.activations)(observations, actions)
        return critic1, critic2
