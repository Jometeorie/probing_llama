#!/bin/bash

# conflicting knowledge部分各层Vi热力图
# root_path=/home/jtj/probing_llama
root_path=/root/paddlejob/workspace/sunweiwei/rank/probing_llama-master

/usr/local/bin/python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/vi/Llama-2-7b-chat-hf.yaml --is_probing True --is_record_all_vi True --root_path ${root_path} --fact_idx -1
/usr/local/bin/python ${root_path}/code/interpret_context_knowledge.py \
    --config_yaml ${root_path}/experiments/vi/Llama-2-13b-chat-hf.yaml --is_probing True --is_record_all_vi True --root_path ${root_path} --fact_idx -1
# /usr/local/bin/python ${root_path}/code/interpret_context_knowledge.py \
#     --config_yaml ${root_path}/experiments/vi/Llama-2-70b-chat-hf.yaml --is_probing True --is_record_all_vi True --root_path ${root_path} --fact_idx -1