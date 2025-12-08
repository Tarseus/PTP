export CUDA_VISIBLE_DEVICES=6

nohup python -u -m ptp_discovery.run_llm_search \
    --generations 5 \
    --population 8 \
    --hf-steps 500 \
    --train-size 20 \
    --valid-sizes 20 \
    --device cuda > nohup.out 2>&1 &