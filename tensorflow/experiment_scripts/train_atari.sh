CUDA_VISIBLE_DEVICES="0" python train.py --env=GopherNoFrameskip-v4 --agent=DQN --double_q=1 --dueling=0 --num_steps=200000000 --buffer_size=1000000
