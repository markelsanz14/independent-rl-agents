CUDA_VISIBLE_DEVICES="0" python train.py --env=BreakoutNoFrameskip-v4 --agent=DQN --double_q=0 --dueling=0 --num_steps=200000000 --buffer_size=1000000 --use_dataset_buffer=1
