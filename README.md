# independent-rl-agents

This repository is intended to have standalone Deep Reinforcement Learning algorithms 
that anyone can download, copy, modify and play with, without much overhead. It is not 
planned to be an all-in one package nor a pip installable one. It will include some 
ways to evaluate the algorithms on different OpenAI Gym environemnts to see their 
performance, but they will not be integrated as part of the algorithms. The main idea 
is to have algorithms that are independent from the application and easy to adapt to 
new environemnts.

### Current status:
| Algorithm                | Status                  | Type of Algorithm                    | Action Space | Paper Link |
|--------------------------|-------------------------|--------------------------------------|--------------|------------|
| DQN                      | :x: Not implemented yet | Model-Free, Off-Policy, Value-Based  | Discrete     |            |
| Doube-DQN                | :x: Not implemented yet | Model-Free, Off-Policy, Value-Based  | Discrete     |            |
| Dueling-DQN              | :x: Not implemented yet | Model-Free, Off-Policy, Value-Based  | Discrete     |            |
| Simplest Policy Gradient | :x: Not implemented yet | Model-Free, On-Policy, Policy-Based  | Both         |            |
| DDPG                     | :white_check_mark: Done | Model-Free, Off-Policy, Actor-Critic | Continuous   |            |
| SAC                      | Planned for future      | Model-Free, Off-Policy, Actor-Critic | Both         |            |
