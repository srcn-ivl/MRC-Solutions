cd ..

nohup python infer_dump_entities_rec.py \
    --vl-base ./sln/Qwen2.5-VL-32B-Instruct\
    --image-base ./sln/VQA_SA\
    --dump-path ./output/vqasa/entities\
    --device auto\
    --max-output-length 512\
    >logs/infer_dump_entities_rec_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

