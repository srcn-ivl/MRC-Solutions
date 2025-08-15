cd ..

CUDA_VISIBLE_DEVICES=0,2 \
    nohup python infer_dump_ovis2_patch.py \
        --vl-base ./sln/Ovis2-34B\
        --image-base ./sln/VQA_SA\
        --question-json ./sln/VQA_SA/VQA-SA-question-completed.json\
        --base-qa ./sln/VQA_SA/0804_VQASA_OV_ZH.json\
        --dump-dir ./output/vqasa/ovis2_mr_patches\
        --device auto\
        --max-output-length 1024\
        >logs/infer_dump_ovis2_patch_$(date +"%Y-%m-%d_%H-%M-%S").log 2>&1 &

