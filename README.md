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
| DQN                      | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |
| Doube-DQN                | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12389/11847) |
| Dueling-DQN              | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     |  |
| Double-Dueling-DQN       | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://arxiv.org/pdf/1511.06581.pdf) |
| Simplest Policy Gradient | :x: Not implemented yet           | Model-Free, On-Policy, Policy-Based  | Both         |  |
| DDPG                     | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Actor-Critic | Continuous   | [Paper](https://arxiv.org/pdf/1509.02971.pdf) |
| SAC                      | Planned for future                | Model-Free, Off-Policy, Actor-Critic | Both         | [Paper](https://arxiv.org/pdf/1812.05905.pdf) |


### Algorithm implementation performance on Atari games:
We compare the performance of the algorithms as reported in the double dqn paper ([wang et al. 2015](https://arxiv.org/pdf/1511.06581.pdf)) with ours for the different Atari Games.

This table shows the original performance vs ours, in that order.

| Game            | DQN              | Double DQN      | Dueling DQN  | Double Dueling DQN |
|-----------------|------------------|-----------------|--------------|--------------------|
| Alien           | 1,620.0 vs       | 3,747.7 vs      |              | 4,461.4 vs         | 
| Amidar          | 978.0 vs         | 1,793.3 vs      |              | 2,354.5 vs         | 
| Assault         | 4,280.4 vs       | 5,393.2 vs      |              | 4,621.0 vs         | 
| Asterix         | 4,359.0 vs       | 17,356.5 vs     |              | 28,188.0 vs        |
| Asteroids       | 1,364.5 vs       | 734.7 vs        |              | 2,837.7 vs         |
| Atlantis        | 279,987 vs       | 106,056 vs      |              | 382,572 vs         |
| Bank Heist      | 455.0 vs         | 1,030.6 vs      |              | 1,611.9 vs         |

