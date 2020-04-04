CUDA_VISIBLE_DEVICES="0" python train.py --env=BreakoutNoFrameskip-v4 --agent=DQN --double_q=1 --dueling=1 --num_steps=200000000 --buffer_size=1000000
