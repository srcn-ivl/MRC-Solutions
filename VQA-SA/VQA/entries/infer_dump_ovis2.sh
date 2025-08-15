cd ..

nohup python infer_dump_ovis2.py \
    --vl-base ./sln/Ovis2-34B\
    --image-base ./sln/VQA_SA\
    --question-json ./sln/VQA_SA/VQA-SA-question.json\
    --dump-dir ./output/vqasa/ovis2_splits_zh\
    --device auto\
    --max-output-length 1024\
    >logs/infer_dump_ovis2_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

