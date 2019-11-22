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

| Game                | DQN              | Double DQN      | Dueling DQN  | Double Dueling DQN |
|---------------------|------------------|-----------------|--------------|--------------------|
| Alien               | 1,620.0 vs       | 3,747.7 vs      |              | 4,461.4 vs         | 
| Amidar              | 978.0 vs         | 1,793.3 vs      |              | 2,354.5 vs         | 
| Assault             | 4,280.4 vs       | 5,393.2 vs      |              | 4,621.0 vs         | 
| Asterix             | 4,359.0 vs       | 17,356.5 vs     |              | 28,188.0 vs        |
| Asteroids           | 1,364.5 vs       | 734.7 vs        |              | 2,837.7 vs         |
| Atlantis            | 279,987 vs       | 106,056 vs      |              | 382,572 vs         |
| Bank Heist          | 455.0 vs         | 1,030.6 vs      |              | 1,611.9 vs         |
| Battle Zone         | 29,900.0 vs      | 31,700.0 vs     |              | 37,150.0 vs        | 
| Beam Rider          | 8,627.5 vs       | 13,772.8 vs     |              | 12,164.0 vs        | 
| Berzerk             | 585.6 vs         | 1,225.4 vs      |              | 1,472.6 vs         | 
| Bowling             | 50.4 vs          | 68.1 vs         |              | 65.5 vs            |
| Boxing              | 88.0 vs          | 91.6 vs         |              | 99.4 vs            |
| Breakout            | 385.5 vs         | 418.5 vs        |              | 345.3 vs           |
| Centipede           | 4,657.7 vs       | 5,409.4 vs      |              | 7,561.4 vs         |
| Chopper Command     | 6,126.0 vs       | 5,809.0 vs      |              | 11,215.0 vs        | 
| Crazy Climber       | 110,763.0 vs     | 117,282.0 vs    |              | 143,570.0 vs       | 
| Defender            | 23,633.0 vs      | 35,338.5 vs     |              | 42,214.0 vs        | 
| Demon Attack        | 12,149.4 vs      | 58,044.2 vs     |              | 60,813.3 vs        |
| Double Dunk         | -6.6 vs          | -5.5 vs         |              | 0.1 vs             |
| Enduro              | 729.0 vs         | 1,211.8 vs      |              | 2,258.2 vs         |
| Fishing Derby       | -4.9 vs          | 15.5 vs         |              | 46.4 vs            |
| Freeway             | 30.8 vs          | 33.3 vs         |              | 0.0 vs             | 
| Frostbite           | 797.4 vs         | 1,683.3 vs      |              | 4,672.8 vs         | 
| Gopher              | 8,777.4 vs       | 14,840.8 vs     |              | 15,718.4 vs        | 
| Gravitar            | 473.0 vs         | 412.0 vs        |              | 588.0 vs           |
| HERO                | 20,437.8 vs      | 20,130.2 vs     |              | 20,818.2 vs        |
| Ice Hockey          | -1.9 vs          | -2.7 vs         |              | 0.5 vs             |
| James Bond          | 768.5 vs         | 1,358.0 vs      |              | 1,312.5 vs         |
| Kangaroo            | 7,259.0 vs       | 12,992.0 vs     |              | 14,854.0 vs        | 
| Krull               | 8,422.3 vs       | 7,920.5 vs      |              | 11,451.9 vs        | 
| Kung-Fu Master      | 26,059.0 vs      | 29,710.0 vs     |              | 34,294.0 vs        | 
| Montezuma's Revenge | 0.0 vs           | 0.0 vs          |              | 0.0 vs             |
| Ms Pac-Man          | 3,085.6 vs       | 2,711.4 vs      |              | 6,283.5 vs         |
| Name This Game      | 8,207.8 vs       | 10,616.0 vs     |              | 11,971.1 vs        |
| Phoenix             | 8,485.2 vs       | 12,252.5 vs     |              | 23,092.2 vs        |
| Pitfall!            | -286.1 vs        | -29.9 vs        |              | 0.0 vs             | 
| Pong                | 19.5 vs          | 20.9 vs         |              | 21.0 vs            | 
| Private Eye         | 146.7 vs         | 129.7 vs        |              | 103.0 vs           | 
| Q*Bert              | 13,117.3 vs      | 15,088.5 vs     |              | 19,220.3 vs        |
| River Raid          | 7,377.6 vs       | 14,884.5 vs     |              | 21,162.6 vs        |
| Road Runner         | 39,544.0 vs      | 44,127.0 vs     |              | 69,524.0 vs        |
| Robotank            | 63.9 vs          | 65.1 vs         |              | 65.3 vs            |
| Seaquest            | 5,860.6 vs       | 16,452.7 vs     |              | 50,254.2 vs        | 
| Skiing              | -13,062.3 vs     | -9,021.8 vs     |              | -8,857.4 vs        | 
| Solaris             | 3,482.8 vs       | 3,067.8 vs      |              | 2,250.8 vs         | 
| Space Invaders      | 1,692.3 vs       | 2,525.5 vs      |              | 6,427.3 vs         |
| Star Gunner         | 54,282.0 vs      | 60,142.0 vs     |              | 89,238.0 vs        |
| Surround            | -5.6 vs          | -2.9 vs         |              | 4.4 vs             |
| Tennis              | 12.2 vs          | -22.8 vs        |              | 5.1 vs             |
| Time Pilot          | 4,870.0 vs       | 8,339.0 vs      |              | 11,666.0 vs        | 
| Tutankham           | 68.1 vs          | 218.4 vs        |              | 211.4 vs           | 
| Up and Down         | 9,989.9 vs       | 22,972.2 vs     |              | 44,939.6 vs        | 
| Venture             | 163.0 vs         | 98.0 vs         |              | 497.0 vs           |
| Video Pinball       | 196,760.4 vs     | 309,941.9 vs    |              | 98,209.5 vs        |
| Wizard Of Wor       | 2,704.0 vs       | 7,492.0 vs      |              | 7,855.0 vs         |
| Yars' Revenge       | 18,098.9 vs      | 11,712.6 vs     |              | 49,622.1 vs        |
| Zaxxon              | 5,363.0 vs       | 10,163.0 vs     |              | 12,944.0 vs        |
