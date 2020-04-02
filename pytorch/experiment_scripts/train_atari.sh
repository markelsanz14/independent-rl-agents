CUDA_VISIBLE_DEVICES="2" python train.py --env=AlienNoFrameskip-v4 --agent=DQN --double_q=1 --dueling=1 --num_steps=100000000 --buffer_size=100000
