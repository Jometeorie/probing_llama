#!/bin/bash

root_path=/home/jtj/probing_llama

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/heatmap/password.yaml --is_probing True --is_plot_heatmap True --root_path ${root_path}