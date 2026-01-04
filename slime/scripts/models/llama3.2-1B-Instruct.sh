MODEL_ARGS=(
   --swiglu
   --num-layers 16
   --hidden-size 2048
   --ffn-hidden-size 8192
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --max-position-embeddings 131072
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-5
   --rotary-base 500000
   --vocab-size 128256
   --kv-channels 64
   --use-rope-scaling
   --rotary-scaling-factor 32.0
)