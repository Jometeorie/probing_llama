#!/bin/bash

# conflicting knowledge部分各层Vi热力图
root_path=/home/jtj/probing_llama

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/heatmap/commonsense.yaml --is_probing True --is_record_vi True --root_path ${root_path} --fact_idx -1