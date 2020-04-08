CUDA_VISIBLE_DEVICES="2" python train.py --env=BreakoutNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=100000000 --buffer_size=100000
