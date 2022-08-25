"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import unfreeze, freeze

import policy
import value_net
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

    new_value, value_info = update_v(target_critic, value, batch, expectile)
    key, rng = jax.random.split(rng)
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, batch, temperature)

    new_critic, critic_info = update_q(critic, new_value, batch, discount)

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
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 finetune: str = 'freeze',
                 encoder: bool = False,
                 rep_module: str = '',
                 opt_decay_schedule: str = "cosine",
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
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            actor_optimiser = optax.chain(optax.scale_by_adam(),
                                optax.scale_by_schedule(schedule_fn))
        else:
            actor_optimiser = optax.adam(learning_rate=actor_lr)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        value_optimiser = optax.adam(learning_rate=value_lr)

        if self.finetune:
            actor_param_labels = freeze({'MLP_0':'rep', 'Dense_0': 'output', 'log_stds': 'output'})
            single_critic_labels = {'MLP_0': {'Dense_0': 'rep', 'Dense_1': 'rep', 'Dense_2': 'output'}}
            critic_param_labels = freeze({'Critic_0': single_critic_labels, 'Critic_1': single_critic_labels})
            value_param_labels = freeze(single_critic_labels)
        
            if self.finetune == 'naive':
                pass
            elif self.finetune == 'freeze':
                pass
            elif self.finetune == 'recuded-lr':
                # TODO: set different lr for head and tail
                pass
            else:
                raise NotImplementedError

            # freeze  representation parameters (i.e. set the gradient of these parameters to zero)
            if self.rep_module == 'backbone':
                actor_optimiser = optax.multi_transform(
                    {'rep': optax.set_to_zero(), 'output': actor_optimiser},
                    actor_param_labels)
                critic_optimiser = optax.multi_transform(
                    {'rep': optax.set_to_zero(), 'output': critic_optimiser},
                    critic_param_labels)
                value_optimiser = optax.multi_transform(
                    {'rep': optax.set_to_zero(), 'output': value_optimiser},
                    value_param_labels)
            elif self.rep_module == 'encoder':
                # TODO
                pass
            else:
                raise NotImplementedError


        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)  # nn.module
        critic = Model.create(critic_def,     # Model (flax.struct.dataclass)
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimiser)

        value_def = value_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=value_optimiser)

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        if encoder:
            # TODO
            pass

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





