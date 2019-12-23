# independent-rl-agents

This repository is intended to have standalone Deep Reinforcement Learning algorithms 
that anyone can download, copy, modify and play with, with very little overhead. It is not 
planned to be an all-in one package nor a pip installable one. It will include some 
ways to evaluate the algorithms on different OpenAI Gym environemnts to see their 
performance, but they will not be integrated as part of the algorithms. The main idea 
is to have algorithms that are independent from the application and easy to adapt to 
new environemnts. Each algorithm contains everything you need in a single file: the neural
network, the replay buffer, helper functions, and the main training function.

All the algorithms have been implemented in both Tensorflow 2.0 and PyTorch. The TF
algorithms use the tf.function decorator for compiling them to graph mode for improved
performance. They also support GPUs for faster training. 

### Current status:
| Algorithm                | Status                            | Type of Algorithm                    | Action Space | Link |
|--------------------------|-----------------------------------|--------------------------------------|--------------|------------|
| DQN                      | :heavy_check_mark: Ready          | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |
| Double-DQN                | :heavy_check_mark: Not Tested     | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12389/11847) |
| Dueling-DQN              | :heavy_check_mark: Not Tested     | Model-Free, Off-Policy, Value-Based  | Discrete     |  |
| Double-Dueling-DQN       | :heavy_check_mark: Not Tested     | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://arxiv.org/pdf/1511.06581.pdf) |
| Simplest Policy Gradient | :x: Not implemented yet           | Model-Free, On-Policy, Policy-Based  | Both         |  |
| DDPG                     | :heavy_check_mark: Not Tested     | Model-Free, Off-Policy, Actor-Critic | Continuous   | [Paper](https://arxiv.org/pdf/1509.02971.pdf) |
| SAC                      | Planned for future                | Model-Free, Off-Policy, Actor-Critic | Both         | [Paper](https://arxiv.org/pdf/1812.05905.pdf) |

### How to use:
The pytroch and tensorflow folderst contain the same code and algorithms.
Inside each of them you will find:
* agents/*: the implementations of the different Deep RL algorithms.
    - agents/dqn.py: The DQN algorithm with both uniform and prioritized experience replay buffers
    , and two types of neural networks (Fully Connected and Convolutional).
    - agents/double_dqn.py: The DoubleDQN algorithm with both uniform and prioritized experience
    replay buffers, and two types of neural networks (Fully Connected and Convolutional).
    - agents/dueling_dqn.py The Dueling DQN version that does not use the double DQN update 
    rule. It contains both the uniform and prioritized experience replay buffers, and two types
    of neural networks (Fully Connected and Convolutional).
    - agents/double_dueling_dqn.py: The Dueling DQN version that uses the double DQN update rule.
    It contains both the uniform and prioritized experience replay buffers, and two types of
    neural networks (Fully Connected and Convolutional).
    - agents/ddpg.py: The DDPG algorithm, with both uniform and prioritized experience replay
    buffers, and two types of neural networks (Fully Connected and Convolutional) for both the
    actor and the critic.
* experiment_scripts/*: scripts to quickly train, evaluate or enjoy a game with an algorithms.
    - experiment_scripts/train_atari.py: Used to train an algorithm on the atari games.
    - experiment_scripts/evaluate_atari.py: Used to evaluate a trained model several times on the
    atari games.
    - experiment_scripts/enjoy_atari.py: Used to play a game with a trained model while rendering
    the game, for visualizing the agent's performance.
* train.py: Main file for training an agent on an environemnt.
* evaluate.py: Main file for evaluating a trained agent on an environment.
* enjoy.py: Main file for visualizing a trained agent interacting with an environment.
* envs.py: Helper file with some lists of environments.
* atari_wrappers.py: File with OpenAI Gym environemnt wrappers that change the functionality of
the environemnts.

### Examples:
To train the DQN algorithm using TensorFlow 2.0 on the Atari Breakout environment for 1,000,000
steps, go to the tensorflow directory and do:
python train.py --env=BreakoutNoFrameskip-v4 --agent=DQN --num_steps=1000000
or modify the experiment_scripts/train_atari.sh file.
To evaluate or enjoy an agent do the same with the evaluate_atari.sh or enjoy_atari.sh scripts.

To get help and see the flags of a file do:
python train.py -h
where train.py is the file you want to see the help for.

### Results:
To see the performance achieved with each algorithm in the different environments
[go to the results section.](https://github.com/markelsanz14/independent-rl-agents/tree/master/results)
