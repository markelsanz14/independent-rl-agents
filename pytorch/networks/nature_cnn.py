import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        """Initializes the neural network."""
        super(NatureCNN, self).__init__()
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
        return self.out(x)
