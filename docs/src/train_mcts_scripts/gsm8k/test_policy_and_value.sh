set -e
export TEST_NO_TERMINAL=1
# export TEST_WITH_TERMINAL=1
# export TEST_COT_GREEDY=1
# export TEST_COT_SC=1

# export CUDA_VISIBLE_DEVICES=4,5,6,7


CT2_DIR={your ct2 model cache}
CRITIC_PATH={your critic model cache}
torchrun --nproc_per_node=8 --master-port 29503 ../../tsllm/offline_rl/test_sft_and_v.py \
    --ct2_dir $CT2_DIR \
    --critic_model_path $CRITIC_PATH \
    --tokenizer_path $CRITIC_PATH \
    --save_dir $1/pi_sftep3_v_sftep1 \
    --env_name gsm8k \
    --test True
