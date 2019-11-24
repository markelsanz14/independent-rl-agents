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
| Alien               | 1,620 vs         | 3,747 vs        |              | 4,461 vs           | 
| Amidar              | 978 vs           | 1,793 vs        |              | 2,354 vs           | 
| Assault             | 4,280 vs         | 5,393 vs        |              | 4,621 vs           | 
| Asterix             | 4,359 vs         | 17,356 vs       |              | 28,188 vs          |
| Asteroids           | 1,364 vs         | 734 vs          |              | 2,837 vs           |
| Atlantis            | 279,987 vs       | 106,056 vs      |              | 382,572 vs         |
| Bank Heist          | 455 vs           | 1,030 vs        |              | 1,611 vs           |
| Battle Zone         | 29,900 vs        | 31,700 vs       |              | 37,150 vs          | 
| Beam Rider          | 8,627 vs         | 13,772 vs       |              | 12,164 vs          | 
| Berzerk             | 585 vs           | 1,225 vs        |              | 1,472 vs           | 
| Bowling             | 50 vs            | 68.1 vs         |              | 65.5 vs            |
| Boxing              | 88 vs            | 91.6 vs         |              | 99.4 vs            |
| Breakout            | 385 vs           | 418 vs          |              | 345 vs             |
| Centipede           | 4,657 vs         | 5,409 vs        |              | 7,561 vs           |
| Chopper Command     | 6,126 vs         | 5,809 vs        |              | 11,215 vs          | 
| Crazy Climber       | 110,763 vs       | 117,282 vs      |              | 143,570 vs         | 
| Defender            | 23,633 vs        | 35,338 vs       |              | 42,214 vs          | 
| Demon Attack        | 12,149 vs        | 58,044 vs       |              | 60,813 vs          |
| Double Dunk         | -6.6 vs          | -5.5 vs         |              | 0.1 vs             |
| Enduro              | 729 vs           | 1,211 vs        |              | 2,258 vs           |
| Fishing Derby       | -4.9 vs          | 15.5 vs         |              | 46.4 vs            |
| Freeway             | 30.8 vs          | 33.3 vs         |              | 0.0 vs             | 
| Frostbite           | 797 vs           | 1,683 vs        |              | 4,672 vs           | 
| Gopher              | 8,777 vs         | 14,840 vs       |              | 15,718 vs          | 
| Gravitar            | 473 vs           | 412 vs          |              | 588 vs             |
| HERO                | 20,437 vs        | 20,130 vs       |              | 20,818 vs          |
| Ice Hockey          | -1.9 vs          | -2.7 vs         |              | 0.5 vs             |
| James Bond          | 768 vs           | 1,358 vs        |              | 1,312 vs           |
| Kangaroo            | 7,259 vs         | 12,992 vs       |              | 14,854 vs          | 
| Krull               | 8,422 vs         | 7,920 vs        |              | 11,451 vs          | 
| Kung-Fu Master      | 26,059 vs        | 29,710 vs       |              | 34,294 vs          | 
| Montezuma's Revenge | 0.0 vs           | 0.0 vs          |              | 0.0 vs             |
| Ms Pac-Man          | 3,085 vs         | 2,711 vs        |              | 6,283 vs           |
| Name This Game      | 8,207 vs         | 10,616 vs       |              | 11,971 vs          |
| Phoenix             | 8,485 vs         | 12,252 vs       |              | 23,092 vs          |
| Pitfall!            | -286 vs          | -29.9 vs        |              | 0.0 vs             | 
| Pong                | 19.5 vs          | 20.9 vs         |              | 21.0 vs            | 
| Private Eye         | 146 vs           | 129 vs          |              | 103 vs             | 
| Q*Bert              | 13,117 vs        | 15,088 vs       |              | 19,220 vs          |
| River Raid          | 7,377 vs         | 14,884 vs       |              | 21,162 vs          |
| Road Runner         | 39,544 vs        | 44,127 vs       |              | 69,524 vs          |
| Robotank            | 63.9 vs          | 65.1 vs         |              | 65.3 vs            |
| Seaquest            | 5,860 vs         | 16,452 vs       |              | 50,254 vs          | 
| Skiing              | -13,062 vs       | -9,021 vs       |              | -8,857 vs          | 
| Solaris             | 3,482 vs         | 3,067 vs        |              | 2,250 vs           | 
| Space Invaders      | 1,692 vs         | 2,525 vs        |              | 6,427 vs           |
| Star Gunner         | 54,282 vs        | 60,142 vs       |              | 89,238 vs          |
| Surround            | -5.6 vs          | -2.9 vs         |              | 4.4 vs             |
| Tennis              | 12.2 vs          | -22.8 vs        |              | 5.1 vs             |
| Time Pilot          | 4,870 vs         | 8,339 vs        |              | 11,666 vs          | 
| Tutankham           | 68.1 vs          | 218 vs          |              | 211 vs             | 
| Up and Down         | 9,989 vs         | 22,972 vs       |              | 44,939 vs          | 
| Venture             | 163 vs           | 98.0 vs         |              | 497 vs             |
| Video Pinball       | 196,760 vs       | 309,941 vs      |              | 98,209 vs          |
| Wizard Of Wor       | 2,704 vs         | 7,492 vs        |              | 7,855 vs           |
| Yars' Revenge       | 18,098 vs        | 11,712 vs       |              | 49,622 vs          |
| Zaxxon              | 5,363 vs         | 10,163 vs       |              | 12,944 vs          |
