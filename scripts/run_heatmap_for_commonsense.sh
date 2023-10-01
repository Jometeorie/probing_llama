#!/bin/bash

# conflicting knowledge部分各层Vi热力图
# 由于要存储的state过多（num of facts * hidden step * layer * d），且需要执行num of facts * hidden step * layer次探针任务，因此速度相对较慢
root_path=/home/jtj/probing_llama
n=5 # 可以根据实际情况选择最好的典型样例用于论文，确定好fact id后，需要根据该fact id生成更多的实例用于更精确的探针

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/heatmap/commonsense.yaml --is_probing True --is_plot_heatmap True --root_path ${root_path} --fact_idx $n