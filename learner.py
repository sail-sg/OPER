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
    # new_target_params = jax.tree_multimap(
    new_target_params = jax.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@jax.jit
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    key0, key1, key2, rng = jax.random.split(rng, 4)
    new_value, value_info = update_v(key0, target_critic, value, batch, expectile)
    new_actor, actor_info = awr_update_actor(key1, actor, target_critic,
                                             new_value, batch, temperature)

    new_critic, critic_info = update_q(key2, critic, new_value, batch, discount)

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
                 finetune: str = '',
                 retrain: str = '',
                 encoder: bool = False,
                 rep_module: str = '',
                 opt_decay_schedule: str = "cosine",
                 last_layer_norm: bool = False,
                 batch_norm: bool = False,
                 max_gradient_norm: float = 1.0,
                 **kwargs):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.finetune = finetune
        self.rep_module = rep_module

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, \
        dropout_key1, dropout_key2, dropout_key3 = jax.random.split(rng, 7)


        if encoder:
            def encoder_generator():
                return Encoder(encoder_hidden_dims, embedding_dim, last_layer_norm=last_layer_norm, batch_norm=batch_norm)
        else:
            encoder_generator = None # encoder are implicitly included in actor/critic hidden layers
        
        if not self.finetune or self.finetune == 'none':
            if opt_decay_schedule == "cosine":
                # negative lr for scale tranformation
                # schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
                # actor_optimiser = optax.chain(optax.scale_by_adam(),
                #                     optax.scale_by_schedule(schedule_fn))

                schedule_fn = optax.cosine_decay_schedule(actor_lr, max_steps)
                actor_optimiser = optax.adamw(schedule_fn)
                                    
            else:
                actor_optimiser = optax.adamw(learning_rate=actor_lr)
            critic_optimiser = optax.adamw(learning_rate=critic_lr)
            value_optimiser = optax.adamw(learning_rate=value_lr)


        else:
            
            # 0.1x learning rate and warmup
            if opt_decay_schedule == "cosine":
                # schedule_fn = optax.cosine_decay_schedule(actor_lr*0.1, max_steps)
                schedule_fn = optax.warmup_cosine_decay_schedule(
                    init_value=actor_lr*0.001, 
                    peak_value=actor_lr*0.1, 
                    warmup_steps=int(max_steps/20),
                    decay_steps=max_steps)
                actor_optimiser = optax.adam(schedule_fn)
            else:
                actor_optimiser = optax.adam(learning_rate=actor_lr*0.1)
            critic_optimiser = optax.adam(learning_rate=critic_lr*0.1)
            value_optimiser = optax.adam(learning_rate=value_lr*0.1)
            
            # actor_param_labels = freeze({'MLP_0':'rep', 'Dense_0': 'pred', 'log_stds': 'pred'})
            # single_critic_labels = {'MLP_0': {'Dense_0': 'rep', 'Dense_1': 'rep', 'Dense_2': 'pred'}}
            # critic_param_labels = freeze({'Critic_0': single_critic_labels, 'Critic_1': single_critic_labels})
            # value_param_labels = freeze(single_critic_labels)
            if len(hidden_dims) > 0:
                actor_param_labels = freeze({
                    'Encoder_0': 'rep', 'MLP_0': 'pred', 'Dense_0': 'pred', 'log_stds': 'pred'})
            else:
                actor_param_labels = freeze({
                    'Encoder_0': 'rep', 'Dense_0': 'pred', 'log_stds': 'pred'})
            single_critic_labels = {'Encoder_0': 'rep', 'MLP_0': 'pred'}
            critic_param_labels = freeze({'Critic_0': single_critic_labels, 'Critic_1': single_critic_labels})
            value_param_labels = freeze(single_critic_labels)
            if self.finetune == 'freeze':
                actor_optimiser2 = optax.set_to_zero()
                critic_optimiser2 = optax.set_to_zero()
                value_optimiser2 = optax.set_to_zero()
            elif self.finetune == 'reduced-lr':
                actor_optimiser2 = optax.adam(learning_rate=actor_lr*0.01)
                critic_optimiser2 = optax.adam(learning_rate=critic_lr*0.01)
                value_optimiser2 = optax.adam(learning_rate=value_lr*0.01)
            else:
                raise NotImplementedError
        

            if retrain == 'pred':
                actor_optimiser = optax.multi_transform(
                    {'rep': actor_optimiser2, 'pred': actor_optimiser},
                    actor_param_labels)
                critic_optimiser = optax.multi_transform(
                    {'rep': critic_optimiser2, 'pred': critic_optimiser},
                    critic_param_labels)
                value_optimiser = optax.multi_transform(
                    {'rep': value_optimiser2, 'pred': value_optimiser},
                    value_param_labels)
            elif retrain == 'repr':
                actor_optimiser = optax.multi_transform(
                    {'rep': actor_optimiser, 'pred': actor_optimiser2},
                    actor_param_labels)
                critic_optimiser = optax.multi_transform(
                    {'rep': critic_optimiser, 'pred': critic_optimiser2},
                    critic_param_labels)
                value_optimiser = optax.multi_transform(
                    {'rep': value_optimiser, 'pred': value_optimiser2},
                    value_param_labels)
            else:
                raise NotImplementedError

        # clip gradient
        actor_optimiser = optax.chain(optax.clip_by_global_norm(max_gradient_norm), actor_optimiser)
        critic_optimiser = optax.chain(optax.clip_by_global_norm(max_gradient_norm), critic_optimiser)
        value_optimiser = optax.chain(optax.clip_by_global_norm(max_gradient_norm), value_optimiser)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            encoder=encoder_generator,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)
        critic_def = value_net.DoubleCritic(hidden_dims, encoder=encoder_generator)  # nn.module
        value_def = value_net.ValueCritic(hidden_dims, encoder=encoder_generator)
        # print(jax.tree_map(lambda layer_params: layer_params.shape, actor_def.param))
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
            batch, self.discount, self.tau, self.expectile, self.temperature)

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
        
    def reinitialize_output_layer(self):
        raise NotImplementedError





