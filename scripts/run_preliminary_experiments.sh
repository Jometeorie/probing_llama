#!/bin/bash
root_path=/root/paddlejob/workspace/sunweiwei/rank/probing_llama-master
# root_path=/home/jtj/probing_llama

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/commonsense/Llama-2-7b-chat-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}
python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/commonsense/Llama-2-13b-chat-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}
# python ${root_path}/code/interpret_context_knowledge.py \
#     --config_yaml ${root_path}/experiments/commonsense/Llama-2-70b-chat-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/password/Llama-2-7b-chat-hf.yaml --is_record_acc True --root_path ${root_path}
python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/password/Llama-2-13b-chat-hf.yaml --is_record_acc True --root_path ${root_path}
# python ${root_path}/code/interpret_context_knowledge.py \
#     --config_yaml ${root_path}/experiments/password/Llama-2-70b-chat-hf.yaml --is_record_acc True --root_path ${root_path}

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/commonsense/Llama-2-7b-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}
python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/commonsense/Llama-2-13b-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}
# python ${root_path}/code/interpret_context_knowledge.py \
#     --config_yaml ${root_path}/experiments/commonsense/Llama-2-70b-hf.yaml --is_record_acc True --fact_idx -1 --root_path ${root_path}

python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/password/Llama-2-7b-hf.yaml --is_record_acc True --root_path ${root_path}
python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/password/Llama-2-13b-hf.yaml --is_record_acc True --root_path ${root_path}
# python ${root_path}/code/interpret_context_knowledge.py \
#     --config_yaml ${root_path}/experiments/password/Llama-2-70b-hf.yaml --is_record_acc True --root_path ${root_path}
