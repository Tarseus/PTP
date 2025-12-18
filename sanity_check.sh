export TZ=Asia/Shanghai
export CUDA_VISIBLE_DEVICES=4,5,6,7

nohup python -m ptp_discovery.sanity_check_free_loss_eval \
    --config configs/free_loss_discovery.yaml \
    --num-candidates 8 \
    --empty-cache > san_check.out 2>&1 &