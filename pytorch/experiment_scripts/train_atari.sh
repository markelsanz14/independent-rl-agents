CUDA_VISIBLE_DEVICES="1" python train.py --env=PongNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=50000000 --buffer_size=100000
