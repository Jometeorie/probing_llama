#!/bin/bash

# conflicting knowledge部分各层Vi热力图
root_path=/home/jtj/probing_llama

for (( n=0; n<=10; n++ ))
do
    python ${root_path}/code/interpret_context_knowledge.py \
        --config_yaml ${root_path}/experiments/vi_with_irr_evidence/Llama-2-7b-chat-hf.yaml --is_probing True --is_record_last_vi True --root_path ${root_path} --fact_idx -1 --num_of_irrelevant_evidence $n
    # python ${root_path}/code/interpret_context_knowledge.py \
    #     --config_yaml ${root_path}/experiments/vi_with_irr_evidence/Llama-2-13b-chat-hf.yaml --is_probing True --is_record_last_vi True --root_path ${root_path} --fact_idx -1 --num_of_irrelevant_evidence $n
    # python ${root_path}/code/interpret_context_knowledge.py \
    #     --config_yaml ${root_path}/experiments/vi_with_irr_evidence/Llama-2-70b-chat-hf.yaml --is_probing True --is_record_last_vi True --root_path ${root_path} --fact_idx -1 --num_of_irrelevant_evidence $n
done