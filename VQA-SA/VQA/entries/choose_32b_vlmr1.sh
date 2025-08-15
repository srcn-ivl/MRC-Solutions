cd ..

nohup python choose_32b_vlmr1.py \
    >logs/choose_32b_vlmr1_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

