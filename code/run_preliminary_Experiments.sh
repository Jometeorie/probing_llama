for (( n=0 ; n<=49 ; n++ ))
do
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-7b-chat-hf.yaml --fact_idx $n
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-13b-chat-hf.yaml --fact_idx $n
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-70b-chat-hf.yaml --fact_idx $n
done

for (( n=0 ; n<=49 ; n++ ))
do
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-7b-hf.yaml --fact_idx $n
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-13b-hf.yaml --fact_idx $n
    python interpret_with_counterfact.py --config_yaml /home/yuanxinwei/tmp/jtj/probing_llama/experiments/process/process_Llama-2-70b-hf.yaml --fact_idx $n
done