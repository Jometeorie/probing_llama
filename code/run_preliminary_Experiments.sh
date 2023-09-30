#!/bin/bash
root_path=/root/paddlejob/workspace/sunweiwei/rank/probing_llama-master

for (( n=0 ; n<=49 ; n++ ))
do
    python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-7b-chat-hf.yaml --fact_idx $n --root_path ${root_path}
   python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-13b-chat-hf.yaml --fact_idx $n --root_path ${root_path}
#    python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-70b-chat-hf.yaml --fact_idx $n --root_path ${root_path}
done

python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-7b-chat-hf.yaml --root_path ${root_path}
python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-13b-chat-hf.yaml --root_path ${root_path}
#python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-70b-chat-hf.yaml --root_path ${root_path}

for (( n=0 ; n<=49 ; n++ ))
do
    python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-7b-hf.yaml --fact_idx $n --root_path ${root_path}
   python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-13b-hf.yaml --fact_idx $n --root_path ${root_path}
#    python interpret_with_counterfact.py --config_yaml ${root_path}/experiments/commonsense/process/process_Llama-2-70b-hf.yaml --fact_idx $n --root_path ${root_path}
done

python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-7b-hf.yaml --root_path ${root_path}
python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-13b-hf.yaml --root_path ${root_path}
#python interpret_with_password.py --config_yaml ${root_path}/experiments/password/process/process_Llama-2-70b-hf.yaml --root_path ${root_path}
