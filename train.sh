#!/bin/bash

tasks="detection-selection-generation"
directory="t5"
version="headstart"
dataroot="data"
params="params.json"

# (Distributed) GPU support
export CUDA_VISIBLE_DEVICES=0
num_gpus=1
if [ ${num_gpus} = 1 ]; then
  python="python3"
else
  python="python3 -m torch.distributed.launch --nproc_per_node ${num_gpus}"
fi

${python} run.py --directory ${directory} \
  --params_file ${directory}/configs/${tasks}/${params} \
  --dataroot ${dataroot} \
  --exp_name ${tasks}-${version} \
  --tfidf
