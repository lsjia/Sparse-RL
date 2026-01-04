# Sparse-RL: Breaking the Memory Wall in Reinforcement Learning via Stable Sparse Rollouts

This repository contains the implementation of **Sparse-RL**, a framework which empowers stable RL training under sparse rollouts. It is built upon customized versions of **SGLang** (for KV-cache compression method) and **Slime** (for the RL training).

## Installation

It is recommended to use docker image according to the slime documentation. The sglang version used in our project is 0.5.5.post1.

### Pull and Start Docker Container

```bash
# Pull the latest image
docker pull slimerl/slime:nightly-dev-20251117b

# Start the container
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it slimerl/slime:latest /bin/bash
```

### Install SGLang-KVC

We need to uninstall the existing sglang package in the image and install our sglang-kvc.

```bash
cd sglang-kvc
pip install -e ".[all]"
cd ..

```

### Install Slime
Slime is already installed in the docker image. To update to the latest verison:
```bash
# Path can be adjusted according to actual situation
cd /root/slime
git pull
pip install -e .
```

### Install Evaluation Dependencies

If you plan to run the evaluation benchmarks, install the additional requirements:

```bash
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt
cd ..
```

## Dataset
The training dataset for Qwen models is stored in `dataset/simplelr_qwen_level3to5/train.jsonl` and dataset for Llama3.2-1B-Instruct is in `dataset/simplelr_level3to5/train.jsonl`.


## Training
We conduct the GRPO training of Qwen2.5-1.5B/3B/7B and llama3.2-1B-Instruct on 4x H20-141G GPUs.

### Model Weight Conversion
We use Megatron as the training backend. Referring to the official Slime documentation, we need to convert Hugging Face format model weights to Megatron torch_dist format.
```bash
cd slime
bash myscripts/hf2megatron.sh
```

### RL Training
You can use the provided shell scripts in `slime/scripts/` to start training.
Here are examples to train a Qwen-2.5-3B model:
- Qwen-2.5-3B with dense rollouts
```bash
cd slime
# Run the training script (Ensure you have sufficient GPU memory)
bash scripts/run-qwen2.5-3b.sh
```

- Qwen-2.5-3B with sparse rollouts (R-KV)
```bash
cd slime
# Run the training script (Ensure you have sufficient GPU memory)
bash scripts/run-qwen2.5-3b-rkv.sh
```

### Arguments

The following in scripts are newly added arguments related to SGLang-KVC:

* `--sglang-compress-algorithm`: Compression algorithm to use, options are `RKV` and `snapkv`.
* `--sglang-compress-max-prompt`: Maximum length before triggering compression.
* `--sglang-compress-divide-method`: ONLY for RKV: compression methods during decode steps, options are `newline` and `step_length`.
* `--sglang-compress-divide-length`: ONLY for RKV with step length: the compression algorithm will execute compression every compress_divide_length steps.

Noted that if KV compression is used, the `--sglang-disable-cuda-graph` argument is required and `--sglang-attention-backend` should be set to `fa3`.

## Evaluation

### Model Weight Conversion
Use the following script to convert the saved Megatron checkpoints back to Hugging Face format:
```bash
cd slime
bash myscripts/megatron2hf.sh
```

### Evaluation

The evaluation code is in `evaluation/simplelr_math_eval`. We use sglang server as the inference backend.

Start the sglang server:
```bash
python3 -m sglang.launch_server --model-path /public/to/model --host 0.0.0.0
```
Execute the evaluation in another terminal:
```bash
cd evaluation/simplelr_math_eval
bash eval.sh
```


## Acknowledgements
We thank the authors of [SGLang](https://github.com/sgl-project/sglang) and [Slime](https://github.com/THUDM/Slime) for their great work, and author of [R-KV](https://github.com/Zefan-Cai/R-KV) for their implementation of the compression algorithm.
