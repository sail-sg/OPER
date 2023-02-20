"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple
import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import unfreeze, freeze

import policy
import value_net
from encoder import Encoder
from actor import update as awr_update_actor
from common import Batch, InfoDict, Model, PRNGKey
from critic import update_q, update_v


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, 
    reweight_eval, reweight_improve, reweight_constraint
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    key0, key1, key2, rng = jax.random.split(rng, 4)
    new_value, value_info = update_v(key0, target_critic, value, batch, expectile, reweight_eval)
    new_actor, actor_info = awr_update_actor(key1, actor, target_critic,
                                             new_value, batch, temperature, reweight_improve, reweight_constraint)

    new_critic, critic_info = update_q(key2, critic, new_value, batch, discount, reweight_eval)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reweight_eval,
                 reweight_improve,
                 reweight_constraint,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 encoder_hidden_dims = None,  # encoder = encoder_hidden_dim + embedding_dim
                 embedding_dim = None,
                 hidden_dims = None,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 retrain: str = '',
                 encoder: bool = False,
                 opt_decay_schedule: str = "cosine",
                 last_layer_norm: bool = False,
                 batch_norm: bool = False,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.reweight_eval = reweight_eval
        self.reweight_improve = reweight_improve
        self.reweight_constraint = reweight_constraint

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, \
        dropout_key1, dropout_key2, dropout_key3 = jax.random.split(rng, 7)

        if encoder:
            def actor_encoder_generator():
                return Encoder(encoder_hidden_dims, embedding_dim, dropout_rate=dropout_rate, last_layer_norm=last_layer_norm, batch_norm=batch_norm)
            def critic_encoder_generator():
                return Encoder(encoder_hidden_dims, embedding_dim, last_layer_norm=last_layer_norm, batch_norm=batch_norm)
        else:
            actor_encoder_generator = None # encoder are implicitly included in actor/critic hidden layers
            critic_encoder_generator = None # encoder are implicitly included in actor/critic hidden layers
        
        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(actor_lr, max_steps)
            actor_optimiser = optax.adam(schedule_fn)
                                
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        value_optimiser = optax.adam(learning_rate=value_lr)

        action_dim = actions.shape[-1]
        # dropout only is added to hidden layers of actor
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            encoder=actor_encoder_generator,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        critic_def = value_net.DoubleCritic(hidden_dims, encoder=critic_encoder_generator)  # nn.module
        value_def = value_net.ValueCritic(hidden_dims, encoder=critic_encoder_generator)
        actor = Model.create(actor_def,
                             inputs=[{'params': actor_key, 'dropout': dropout_key1}, observations],
                             tx=actor_optimiser)

        critic = Model.create(critic_def,     # Model (flax.struct.dataclass)
                              inputs=[{'params': critic_key, 'dropout': dropout_key2}, observations, actions],
                              tx=critic_optimiser)

        value = Model.create(value_def,
                             inputs=[{'params': value_key, 'dropout': dropout_key3}, observations],
                             tx=value_optimiser)

        target_critic = Model.create(
            critic_def, inputs=[{'params': critic_key, 'dropout': dropout_key2}, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch, self.discount, self.tau, self.expectile, self.temperature, 
            self.reweight_eval, self.reweight_improve, self.reweight_constraint)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    def save(self, dir):
        # not save optimizer
        self.critic.save(dir / 'critic.ckpt')
        self.target_critic.save(dir / 'target_critic.ckpt')
        self.actor.save(dir / 'actor.ckpt')
        self.value.save(dir / 'value.ckpt')

    def load(self, dir):
        self.critic = self.critic.load(dir / 'critic.ckpt')
        self.target_critic = self.target_critic.load(dir / 'target_critic.ckpt')
        self.actor = self.actor.load(dir / 'actor.ckpt')
        self.value = self.value.load(dir / 'value.ckpt')





