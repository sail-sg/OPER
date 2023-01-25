import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		self.log_std = nn.Parameter(torch.zeros(action_dim))
		self.log_std.requires_grad = True
		self.max_action = max_action
		
	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		mu = torch.tanh(self.l3(a))
		log_std = torch.clip(self.log_std, -5, 2)
		# dist = D.Normal(mu, log_std.exp())
		# action = dist.sample().clamp(-1., 1.)

		z = torch.randn_like(mu, requires_grad=False)
		action = z * log_std.exp() + mu

		return action


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3_BC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		reweight_eval,
		reweight_improve,
		reweight_constraint,
		clip_constraint,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		alpha=2.5,
		**kwargs,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.alpha = alpha

		self.reweight_eval = reweight_eval
		self.reweight_improve = reweight_improve
		self.reweight_constraint = reweight_constraint
		self.clip_constraint = clip_constraint

		self.total_it = 0


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()

	def batch_select_action(self, state):
		state = torch.FloatTensor(state).to(device)
		return self.actor(state).cpu().data.numpy()


	def train(self, replay_buffer):
		self.total_it += 1
		# Sample replay buffer 
		state, action, next_state, reward, not_done, weight = replay_buffer.sample()

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target(next_state) + noise
			).clamp(-self.max_action, self.max_action)

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + not_done * self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q, reduction='none') + F.mse_loss(current_Q2, target_Q, reduction='none')
		if self.reweight_eval:
			critic_loss *= weight
		critic_loss = critic_loss.mean()

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Delayed policy updates
		actor_infos = {}
		if self.total_it % self.policy_freq == 0:

			# Compute actor loss
			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			# policy improvement
			actor_loss = Q
			if self.reweight_improve:
				actor_loss *= weight
			actor_loss = actor_loss.mean()
			# policy constraint
			constraint_loss = F.mse_loss(pi, action, reduction='none') 
			if self.reweight_constraint:
				if self.clip_constraint == 1:
					c_weight = torch.clamp(weight, 1.0)
				elif self.clip_constraint == 2:
					c_weight = copy.deepcopy(weight)
					c_weight[weight < 1] = torch.sqrt(weight[weight < 1])
				else:
					c_weight = weight
				constraint_loss *= c_weight
			constraint_loss = constraint_loss.mean()
			actor_loss = -lmbda * actor_loss + constraint_loss
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			# log actor training
			actor_infos = {
				"critic_loss": critic_loss.mean().cpu(),
				"actor_loss": actor_loss.mean().cpu(),
				"constraint_loss": constraint_loss.mean().cpu(),
				"lambda": lmbda.cpu(), 
			}

		return {
			"Q1": current_Q1.mean().cpu(),
			"Q2": current_Q2.mean().cpu(),
			**actor_infos,
		}

	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)
