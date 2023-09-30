#!/bin/bash

root_path=/home/jtj/probing_llama

python interpret_with_password.py --config_yaml ${root_path}/experiments/heatmap/process_password.yaml --save_hidden_states True --root_path ${root_path}
python probing_multiple_steps_with_LR.py --config_yaml ${root_path}/experiments/heatmap/probing_password.yaml --root_path ${root_path}