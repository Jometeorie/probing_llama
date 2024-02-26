#!/bin/bash

# Relatively slow due to the large number of states to be stored (num of facts * hidden step * layer * d) and the need to perform num of facts * hidden step * layer probing tasks
root_path=/home/jtj/probing_llama
n=5 # Case study.

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/heatmap/commonsense.yaml --is_probing True --is_plot_heatmap True --root_path ${root_path} --fact_idx $n