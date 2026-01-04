
PROMPT_TYPE="boxed"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
temperature=0.8
max_tokens=8192
top_p=0.95
benchmarks=gsm8k,math500,gaokao2023en,minerva_math,olympiadbench,aime24,amc23
OVERWRITE=true
N_SAMPLING=1

MODEL_NAME_OR_PATH=/path/to/model
OUTPUT_DIR=/path/to/output
bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR $temperature $max_tokens $top_p $benchmarks $OVERWRITE $N_SAMPLING