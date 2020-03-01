import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CriticNetworkConv(nn.Module):
    """Convolutional neural network for the Atari games."""

    def __init__(self):
        """Initializes the neural network."""
        super(CriticNetworkConv, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv1size = conv2d_size_out(84, kernel_size=8, stride=4)
        conv2size = conv2d_size_out(conv1size, kernel_size=4, stride=2)
        conv3size = conv2d_size_out(conv2size, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * conv3size * conv3size, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten input.
        return self.out(x)


class ActorNetworkConv(nn.Module):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions, min_action_values, max_action_values):
        """Initializes the neural network."""
        super(ActorNetworkConv, self).__init__()
        self.min_action_values = min_action_values
        self.max_action_values = max_action_values
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        conv1size = conv2d_size_out(84, kernel_size=8, stride=4)
        conv2size = conv2d_size_out(conv1size, kernel_size=4, stride=2)
        conv3size = conv2d_size_out(conv2size, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * conv3size * conv3size, 512)
        self.out = nn.Linear(512, num_actions)

    def forward(self, states):
        """Forward pass of the neural network with some inputs.
        Args:
            states: Tensor, batch of states.
        Returns:
            qs: Tensor, the q-values of the given state for all possible
                actions.
        """
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten input.
        x = self.out(x)
        action = x * self.max_action_values
        action_clamped = action.clamp(self.min_action_values, self.max_action_values)
        return action_clamped


class CriticNetwork(nn.Module):
    """Fully connected network for simple games."""

    def __init__(self, num_inputs):
        """Initializes the neural network."""
        super(CriticNetworkConv, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, inputs):
        states, actions = inputs
        x = torch.cat([states, actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class ActorNetwork(nn.Module):
    """Fully connected network for simple games."""

    def __init__(self, num_state_feats, num_action_feats, max_action_values):
        """Initializes the neural network."""
        super(ActorNetworkConv, self).__init__()
        self.max_action_values = max_action_values
        self.fc1 = nn.Linear(num_state_feats, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, num_action_feats)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = F.tanh(self.out(x)) * self.max_action_values
        return action


class DDPG(object):
    def __init__(
        self,
        env_name,
        num_state_feats,
        num_action_feats,
        min_action_values,
        max_action_values,
        replay_buffer=None,
        actor_lr=1e-4,
        critic_lr=1e-4,
        buffer_size=100000,
        discount=0.99,
    ):
        self.num_state_feats = num_state_feats
        self.num_action_feats = num_action_feats
        self.min_action_values = min_action_values
        self.max_action_values = max_action_values

        self.discount = discount
        self.buffer = replay_buffer

        if env_name == "CarRacing-v0":
            self.actor = ActorNetworkConv(
                num_action_feats, min_action_values, max_action_values
            )
            self.actor_target = ActorNetworkConv(
                num_action_feats, min_action_values, max_action_values
            )
            self.critic = CriticNetworkConv()
            self.critic_target = CriticNetworkConv()
        else:
            self.actor = ActorNetwork(
                num_state_feats, num_action_feats, max_action_values
            )
            self.actor_target = ActorNetwork(
                num_state_feats, num_action_feats, max_action_values
            )
            self.critic = CriticNetwork(num_state_feats, num_action_feats)
            self.critic_target = CriticNetwork(num_state_feats, num_action_feats)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss.

        self.save_path = "./saved_models/DQN-{}.pt".format(env_name)
        if os.path.isfile(self.save_path):
            self.main_nn = torch.load(self.save_path)
            print("Loaded model from {}:".format(self.save_path))

    def save_checkpoint(self, step):
        torch.save(self.main_nn, self.save_path)
        print("Saved main_nn checkpoint for step {}: {}".format(step, self.save_path))

    def take_exploration_action(self, state, noise_scale=0.1):
        """Takes action using self.actor network and adding noise for
        exploration."""
        act = self.actor(state) + torch.distributions.Normal(
            loc=(torch.tensor([0.0 for _ in range(self.num_action_feats)])),
            scale=torch.tensor([noise_scale for _ in range(self.num_action_feats)]),
        )
        return act.clamp(self.min_action_values, self.max_action_values)[0]

    def update_target_network(self, source_weights, target_weights, tau=0.001):
        """Updates target networks using Polyak averaging."""
        for source_weight, target_weight in zip(source_weights, target_weights):
            target_weight.data.copy_(tau * source_weight + (1.0 - tau) * target_weight)

    def train_step(
        self, batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    ):
        # Calculate critic target.
        with torch.no_grad():
            next_actions = self.actor_target(batch_next_states)
            next_qs = self.critic_target((batch_next_states, next_actions))
            next_qs = next_qs.view(-1)
            target = batch_rewards + (1.0 - batch_dones) * self.discount * next_qs
        # Calculate critic loss.
        qs = self.critic((batch_states, batch_actions))
        qs = qs.view(-1)
        critic_loss = self.loss_fn(qs, target)
        nn.utils.clip_grad_norm_(critic_loss, max_norm=10)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Calculate actor loss.
        with torch.no_grad():
            policy_actions = self.actor(batch_states)
        policy_qs = self.critic((batch_states, policy_actions))
        actor_loss = -policy_qs.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_network(
            self.critic.named_modules(), self.critic_target.named_modules()
        )
        self.update_target_network(
            self.actor.named_modules(), self.actor_target.named_modules()
        )
        return actor_loss, critic_loss
