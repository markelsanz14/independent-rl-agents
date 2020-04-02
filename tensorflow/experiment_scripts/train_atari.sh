CUDA_VISIBLE_DEVICES="1" python train.py --env=BoxingNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=200000000 --buffer_size=1000000
