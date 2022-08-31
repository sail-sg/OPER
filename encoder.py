from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class Encoder(nn.Module):
    hidden_dims: Sequence[int]
    embedding_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        inputs = observations
        embedding = MLP((*self.hidden_dims, self.embedding_dim),
                     activations=self.activations,
                     activate_final=False)(inputs)
        return embedding

class Dynamic(nn.Module):
    embedding_dim: int
    action_dim: int
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.stack()
        embedding = MLP((*self.hidden_dims, self.embedding_dim),
                     activations=self.activations,
                     activate_final=False)(inputs)
        return embedding