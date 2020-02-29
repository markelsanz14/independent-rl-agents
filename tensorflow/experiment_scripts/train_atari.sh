CUDA_VISIBLE_DEVICES="3" python train.py --env=DemonAttackNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=200000000 --buffer_size=1000000
