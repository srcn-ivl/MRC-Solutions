cd ..

CUDA_VISIBLE_DEVICES=0 \
    nohup python \
    infer_dump_question_comp.py \
    >logs/quescomp_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

