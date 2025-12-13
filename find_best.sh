export TZ=Asia/Shanghai
export CUDA_VISIBLE_DEVICES=5

# PTP-DSL discovery (original pipeline).
# nohup python -u -m ptp_discovery.run_llm_search \
#     --generations 5 \
#     --population 8 \
#     --hf-steps 500 \
#     --train-size 20 \
#     --valid-sizes 20 \
#     --device cuda > nohup_ptp_dsl.out 2>&1 &

# Free-form loss discovery (EoH-style). Uncomment to enable.
nohup python ptp_discovery/run_free_loss_eoh.py \
  --config configs/free_loss_discovery.yaml \
  --device cuda > nohup_free_loss.out 2>&1 &
