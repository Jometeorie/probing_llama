#!/bin/bash

# conflicting knowledge部分各层Vi热力图
# 由于要存储的state过多（num of facts * hidden step * layer * d），且需要执行num of facts * hidden step * layer次探针任务，因此速度相对较慢，之后可以尝试进一步优化
root_path=/home/jtj/probing_llama
n=5 # 可以根据实际情况选择最好的典型样例用于论文，确定好fact id后，需要根据该fact id生成更多的实例用于更精确的探针

python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/heatmap/process_commonsense.yaml --save_hidden_states True --fact_idx $n --root_path ${root_path}
python probing_multiple_steps_with_LR.py --config_yaml ${root_path}/experiments/heatmap/probing_commonsense.yaml --root_path ${root_path}