# independent-rl-agents

This repository is intended to have standalone Deep Reinforcement Learning algorithms 
that anyone can download, copy, modify and play with, without much overhead. It is not 
planned to be an all-in one package nor a pip installable one. It will include some 
ways to evaluate the algorithms on different OpenAI Gym environemnts to see their 
performance, but they will not be integrated as part of the algorithms. The main idea 
is to have algorithms that are independent from the application and easy to adapt to 
new environemnts.

All the algorithms have been implemented using Tensorflow 2.0, but use the tf.function 
decorator for compiling them to graph mode for improved performance.

### Current status:
| Algorithm                | Status                            | Type of Algorithm                    | Action Space | Link |
|--------------------------|-----------------------------------|--------------------------------------|--------------|------------|
| DQN                      | :white_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |
| Doube-DQN                | :white_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12389/11847) |
| Dueling-DQN              | :white_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://arxiv.org/pdf/1511.06581.pdf) |
| Simplest Policy Gradient | :x: Not implemented yet           | Model-Free, On-Policy, Policy-Based  | Both         |  |
| DDPG                     | :white_check_mark: Ready          | Model-Free, Off-Policy, Actor-Critic | Continuous   | [Paper](https://arxiv.org/pdf/1509.02971.pdf) |
| SAC                      | Planned for future                | Model-Free, Off-Policy, Actor-Critic | Both         | [Paper](https://arxiv.org/pdf/1812.05905.pdf) |
