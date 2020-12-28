#!/bin/bash
echo "[!] Make sure your checkpoint has the best validation score!"

# Model setting
tasks="detection-selection"
directory="t5"
version="headstart"
checkpoint="runs/${tasks}-${version}"
params="params.json"

# Data setting
dataroot="data"
split="val"
output_suffix=""

# Prepare directories for intermediate results of each subtask
eval_dir="pred/${split}/${tasks}-${version}"
mkdir -p ${eval_dir}

# (Distributed) GPU support
export CUDA_VISIBLE_DEVICES=0
num_gpus=1
if [ ${num_gpus} = 1 ]; then
  python="python3"
else
  python="python3 -m torch.distributed.launch --nproc_per_node ${num_gpus}"
fi

# Knowledge-seeking turn detection
labels=$([[ ${tasks} =~ "detection" ]] && echo "--no_labels" || echo "")
output_file="${eval_dir}/${tasks}${output_suffix}.json"

${python} run.py --directory ${directory} --eval_only --checkpoint ${checkpoint} \
  --params_file ${directory}/configs/${tasks}/${params} \
  --eval_dataset ${split} \
  --dataroot ${dataroot} \
  --tfidf \
  --output_file ${output_file} \
  --eval_all_snippets \
  --eval_desc ${split} \
  ${labels}

scorefile="${eval_dir}/${task}${output_suffix}.score.json"
python3 scripts/scores.py \
  --dataset ${split} \
  --dataroot ${dataroot} \
  --outfile ${output_file} \
  --scorefile ${scorefile}
