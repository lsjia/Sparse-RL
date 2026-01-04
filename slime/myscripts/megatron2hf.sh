PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
  --input-dir /path/megatron-checkpoint \
  --output-dir /path/hf-checkpoint \
  --origin-hf-dir /path/model