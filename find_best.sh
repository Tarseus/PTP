export CUDA_VISIBLE_DEVICES=6

nohup python -u -m ptp_discovery.run_llm_search \
    --generations 5 \
    --population 8 \
    --hf-steps 1563 \
    --train-size 20 \
    --valid-sizes 100 \
    --device cuda > nohup.out 2>&1 &