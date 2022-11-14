from typing import Tuple
import jax
import jax.numpy as jnp
from common import Batch, InfoDict, Model, Params, PRNGKey
from functools import partial
from jax import jit

def loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def update_v(key: PRNGKey, critic: Model, value: Model, batch: Batch,
             expectile: float, reweight_eval) -> Tuple[Model, InfoDict]:
    actions = batch.actions
    q1, q2 = critic(batch.observations, actions, training=False)
    q = jnp.minimum(q1, q2)
    weights = batch.weights * reweight_eval + jnp.ones_like(batch.weights) * (1-reweight_eval)

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, batch.observations, rngs={'dropout': key})
        value_loss = loss(q - v, expectile) 
        value_loss *= weights
        value_loss = value_loss.mean()
        return value_loss, {
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)

    return new_value, info

# @partial(jit, static_argnums=5)
def update_q(key: PRNGKey, critic: Model, target_value: Model, batch: Batch,
             discount: float, reweight_eval) -> Tuple[Model, InfoDict]:
    key1, key2 = jax.random.split(key, 2)
    next_v = target_value(batch.next_observations, training=False)

    target_q = batch.rewards + discount * batch.masks * next_v
    weights = batch.weights * reweight_eval + jnp.ones_like(batch.weights) * (1-reweight_eval)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply({'params': critic_params}, batch.observations,
                              batch.actions, rngs={'dropout': key2})
        critic_loss = (q1 - target_q)**2 + (q2 - target_q)**2
        critic_loss *= weights
        critic_loss = critic_loss.mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
