# independent-rl-agents

This repository is intended to have standalone Deep Reinforcement Learning algorithms 
that anyone can download, copy, modify and play with, with very little overhead. It is not 
planned to be an all-in-one package or a pip installable one. It includes some 
ways to evaluate the algorithms on different OpenAI Gym environemnts to see their 
performance, but they are separate from the algorithms. The main idea is to have 
algorithms that are independent from the application and easy to adapt to new 
environemnts.

**You will need to select a replay_buffer, a network, and an agent from the different
directories, and they will work together. Scripts are added for quick experimentation.**

All the algorithms have been implemented in both Tensorflow 2.0 and PyTorch (not working yet).
The algorithms have been optimized for performance and support GPUs for faster training.
If additional improvements are possible, please open an issue or submit a pull request.

### Current status:
| Algorithm   | Status                   | Type of Algorithm                    | Action Space | Link       |
|-------------|--------------------------|--------------------------------------|--------------|------------|
| DQN         | :heavy_check_mark: Ready | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) |
| Double-DQN  | :heavy_check_mark: Ready | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12389/11847) |
| Dueling-DQN | :heavy_check_mark: Ready | Model-Free, Off-Policy, Value-Based  | Discrete     | [Paper](https://arxiv.org/pdf/1511.06581.pdf) |
| REINFORCE   | :x: Not implemented yet  | Model-Free, On-Policy, Policy-Based  | Both         |  |
| DDPG        | Not Working              | Model-Free, Off-Policy, Actor-Critic | Continuous   | [Paper](https://arxiv.org/pdf/1509.02971.pdf) |
| SAC         | Planned for future       | Model-Free, Off-Policy, Actor-Critic | Both         | [Paper](https://arxiv.org/pdf/1812.05905.pdf) |

### How to use:
TLDR; take a network and a buffer, and pass them to an agent! As simple as that!

The pytroch and tensorflow folderst contain the same code and algorithms.
Inside each of them you will find:

* agents/*: the implementations of the different Deep RL algorithms. Just pass the networks
and the replay buffer to each of this algorithms, along with the hyperparameters.
    - dqn.py: ready, tested and evaluated.
    - double_dqn.py: ready, and tested.
    - ddpg.py: Not ready.
* networks/*: the different neural networks to choose from.
    - dense.py: a simple NN with dense layers.
    - nature_cnn.py: a convolutional NN, with the same architecture as the original DQN 
    paper published on Nature magazine.
    - dueling_cnn.py: a convolution NN with dueling architecture. Same architecture as 
    the dueling DQN paper.
* replay_buffers/*: the different experience replay buffers to choose from.
    - uniform.py: a replay buffer that uniformly samples random steps. It contains a numpy 
    and a tf.data.Dataset implementation. Sometimes with GPUs, the Dataset one can be faster.
    - prioritized.py: implementation of prioritized experience replay buffer. Not working fast enough.
* experiment_scripts/*: scripts to quickly train, evaluate or enjoy a game with an algorithms.
    - experiment_scripts/train_atari.py: Used to train an algorithm on the atari games.
    - experiment_scripts/evaluate_atari.py: Used to evaluate a trained model several times on the
    atari games.
    - experiment_scripts/enjoy_atari.py: Used to play a game with a trained model while rendering
    the game, for visualizing the agent's performance.
* train.py: Main file for training an agent on an environemnt.
* evaluate.py: Main file for evaluating a trained agent on an environment.
* enjoy.py: Main file for visualizing a trained agent interacting with an environment.

### Examples:
To train the DQN algorithm using TensorFlow 2.0 on the Atari Breakout environment for 1,000,000
steps, go to the tensorflow directory and do:
python train.py --env=BreakoutNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=1000000
or modify the experiment_scripts/train_atari.sh file.
To evaluate or enjoy an agent do the same with the evaluate_atari.sh or enjoy_atari.sh scripts.

To get help and see the flags of a file do:
python train.py -h
where train.py is the file you want to see the help for.

### Results:
To see the performance achieved with each algorithm in the different environments
[go to the results section.](https://github.com/markelsanz14/independent-rl-agents/tree/master/results)
