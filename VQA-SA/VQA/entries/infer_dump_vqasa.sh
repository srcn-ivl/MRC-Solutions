cd ..

nohup python infer_dump_vqasa.py \
    --vl-base ./sln/Qwen2.5-VL-32B-Instruct\
    --image-base ./sln/VQA_SA\
    --dump-path ./output/vqasa/32b_splits\
    --device auto\
    --max-output-length 2048\
    >logs/infer_dump_vqasa_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

