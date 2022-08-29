from typing import Callable, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from common import MLP


class Encoder(nn.module):
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