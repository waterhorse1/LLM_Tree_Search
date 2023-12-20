CONTROLER_PORT=21101
echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)
WORKER_BASE_PORT=30010
MODEL_PATH=/data/ziyu/huggingface_hub_upload/llama2-7b-game24-sft-ep3-hf
VALUE_MODEL_PATH=/data/ziyu/huggingface_hub_upload/llama2-7b-game24-value-sft-ep3/

tmux start-server

tmux new-session -s FastChat -n controller -d
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT}" Enter

echo "Starting Controller..."
echo "Wait 10 seconds ..."
sleep 10

echo "Starting workers"
for i in $(seq 0 7)
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  tmux new-window -n worker_$i
  tmux send-keys "CUDA_VISIBLE_DEVICES=$i $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://localhost:$CONTROLER_PORT --port $WORKER_PORT --worker-address http://localhost:$WORKER_PORT --dtype bfloat16 --swap-space 32 --gpu_memory_utilization 0.6" Enter

  VALUE_WORKER_PORT=$((WORKER_BASE_PORT+i+100))
  tmux new-window -n value_worker_0
  tmux send-keys "CUDA_VISIBLE_DEVICES=$i $PYTHON_EXECUTABLE -m fastchat.serve.value_model_worker --model-path $VALUE_MODEL_PATH --controller-address http://localhost:$CONTROLER_PORT --port $VALUE_WORKER_PORT --worker-address http://localhost:$VALUE_WORKER_PORT" Enter
done

