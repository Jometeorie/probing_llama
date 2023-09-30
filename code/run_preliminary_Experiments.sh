#!/bin/bash
root_path=/root/paddlejob/workspace/sunweiwei/rank/probing_llama-master
# root_path=/home/jtj/probing_llama

python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-7b-chat-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}
# python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-13b-chat-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}
# python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-70b-chat-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}

# python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-7b-chat-hf.yaml --record_acc True --root_path ${root_path}
# #python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-13b-chat-hf.yaml --record_acc True --root_path ${root_path}
# #python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-70b-chat-hf.yaml --record_acc True --root_path ${root_path}

# python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-7b-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}
# python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-13b-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}
# python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-70b-hf.yaml --record_acc True --fact_idx -1 --root_path ${root_path}

# python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-7b-hf.yaml --record_acc True --root_path ${root_path}
# python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-13b-hf.yaml --record_acc True --root_path ${root_path}
# python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-70b-hf.yaml --record_acc True --root_path ${root_path}
