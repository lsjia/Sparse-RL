#!/bin/bash

#for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-3B.sh"

CKPT_ARGS=(
   --hf-checkpoint ~/MODELS/Qwen/Qwen2___5-3B
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load ~/MODELS/Qwen/Qwen2___5-3B_torch_dist
   --load ~/shared_data/simplerl-qwen2.5-3b-rkv/checkpoints
   --save ~/shared_data/simplerl-qwen2.5-3b-rkv/checkpoints
   --save-interval 10
)

ROLLOUT_ARGS=(
   --prompt-data dataset/SimpleRL-Zoo/simplelr_qwen_level3to5/train.jsonl
   --input-key prompt
   --label-key label
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 128
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 1.0

   --global-batch-size 256
   --balance-data
)


PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.0001
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   #--eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project sparserl
   --wandb-group simplerl-qwen2.5-3b-rkv
   --wandb-key WANDB_API_KEY
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.7
   --sglang-disable-cuda-graph
   --sglang-compress-algorithm RKV
   --sglang-compress-max-prompt 512
   --sglang-compress-divide-method step_length
   --sglang-compress-divide-length 128
   --sglang-attention-backend fa3
   --sglang-router-request-timeout-secs 18000
   --sglang-server-concurrency 1024
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
   --rollout-health-check-first-wait 1200.0
   --rollout-health-check-timeout 600.0
)

CUSTOM_ARGS=(
   --use-tis
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 4 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# export WANDB_ID=
# export WANDB_RESUME=allow

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"


ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
