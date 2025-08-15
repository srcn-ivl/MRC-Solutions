cd ..

CUDA_VISIBLE_DEVICES=1 \
    nohup python \
    infer_dump_question_parse.py \
    >logs/quespars_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

