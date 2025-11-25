
import torch
import torch.nn as nn
from transformers import OPTModel, OPTConfig
import numpy as np
import torch.nn.functional as F
import random
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, config_path, num_classes, mlp_hidden_size=768):
        super(Actor, self).__init__()
        config = OPTConfig.from_pretrained(config_path)

        self.opt = OPTModel.from_pretrained(config_path, config=config)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, num_classes)
        )


    def forward(self, input_ids, attention_mask=None):

        outputs = self.opt(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        act_output = self.mlp(last_hidden_state[:, -1, :])

        return act_output


class Critic(nn.Module):
    def __init__(self, config_path, mlp_hidden_size=768):
        super(Critic, self).__init__()

        config = OPTConfig.from_pretrained(config_path)

        self.opt = OPTModel.from_pretrained(config_path, config=config)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_size),
            nn.Tanh(),
            nn.Linear(mlp_hidden_size, 1)
        )
    def forward(self, input_ids, attention_mask=None):

        outputs = self.opt(input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs.last_hidden_state

        value_output = self.mlp(last_hidden_state[:, -1, :])
        return value_output

class PPO:
    def __init__(self, actor, critic, lr_actor=1e-5, lr_critic=1e-6, gamma=0.95, epsilon=0.3):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.epsilon = epsilon
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr_actor, eps=1e-5)
        self.optimizer_critic = torch.optim.Adam(critic.parameters(), lr=lr_critic, eps=1e-5)

    def get_action(self, state, attention_mask):
        with torch.no_grad():
            action_prob = self.actor(state, attention_mask)
        dist = Categorical(logits=action_prob)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action, action_logprob

    def get_action_eval(self, state, attention_mask):
        with torch.no_grad():
            action_prob = self.actor(state, attention_mask)
        action = torch.argmax(action_prob, dim=1)
        return action, action_prob[range(len(action)), action]

    def get_value(self, state, attention_mask):
        with torch.no_grad():
            value_prob = self.critic(state, attention_mask)
        return value_prob

    def get_returns(self, rewards, device):
        rewards = torch.t(rewards)
        returns = torch.zeros_like(rewards, dtype=rewards.dtype, device=device)
        length = rewards.size(-1)
        discounted_reward = torch.tensor(0.0, dtype=rewards.dtype, device=device)
        for t in reversed(range(length)):
            discounted_reward = rewards[:, t] + (self.gamma * discounted_reward)
            returns[:, t] = discounted_reward
        return torch.t(returns)

    def update(self, trajectorys, per_batch_size, sample_size, device, wanddb):

        sample_size = sample_size if sample_size < len(trajectorys) else len(trajectorys)
        for k in range(per_batch_size):
            sampled_trajs = random.sample(trajectorys, sample_size)
            self.optimizer_actor.zero_grad()
            self.optimizer_critic.zero_grad()

            states_bs = []
            rewards_bs = []
            old_action_pros_bs = []
            actions_bs = []
            attention_masks_bs = []
            old_values_bs = []
            for traj in sampled_trajs:
                states_bs.extend(traj['states'])
                rewards_bs.extend([torch.tensor(value).to(device) for value in traj['rewards']])
                old_action_pros_bs.extend(traj['old_action_pros'])
                actions_bs.extend(traj['actions'])
                attention_masks_bs.extend(traj['attention_masks'])
                old_values_bs.extend(traj['old_values'])
            states_bst = torch.stack(states_bs)
            old_action_pros_bst = torch.stack(old_action_pros_bs)
            actions_bst = torch.stack(actions_bs)
            attention_masks_bst = torch.stack(attention_masks_bs)
            old_values_bst = torch.squeeze(torch.stack(old_values_bs), dim=2)

            rewards_bst = torch.stack(rewards_bs)

            new_values = self.critic(states_bst, attention_masks_bst)

            logit = self.actor(states_bst, attention_masks_bst)
            dist = Categorical(logits=logit)
            new_probs = dist.log_prob(actions_bst)

            returns = self.get_returns(rewards_bst, device)
            advantages = returns - old_values_bst

            ratio = torch.exp(new_probs-old_action_pros_bst)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(new_values, returns)
            wanddb.log({"train_actor_loss": actor_loss,
                        "train_critic_loss": critic_loss})
            actor_loss.backward()
            critic_loss.backward()
            self.optimizer_actor.step()
            self.optimizer_critic.step()



if __name__ == '__main__':
    pass