from process_prompt import process_fact_to_prompt
from hidden_states_obj import HiddenStates, HiddenStatesEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import os
import torch
import argparse
import yaml
import json
import jsonlines

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

parser = argparse.ArgumentParser()
parser.add_argument('--config_yaml', type=str)
parser.add_argument('--root_path', type=str, default='/home/jtj/probing_llama')
parser.add_argument('--save_hidden_states', type=bool, default=False)
args = parser.parse_args()
with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

os.environ['CUDA_VISIBLE_DEVICES']=str(config.environment.cuda_visible_devices[0])
results_files = os.listdir(os.path.join(args.root_path, config.data.output_path))
for filename in results_files:
    if '.jsonlines' in filename:
        os.remove(os.path.join(args.root_path, config.data.output_path, filename))
        print('Delete file', os.path.join(args.root_path, config.data.output_path, filename))

facts = []
for label_idx in range(config.data.num_of_labels):
    fact = pd.read_csv(os.path.join(args.root_path, config.data.input_path, 'password_evidence/fact_%s.txt' % label_idx), sep = '------', header = None, engine = 'python')
    fact = fact.reset_index(drop=True)
    facts.append(fact)

tokenizer = AutoTokenizer.from_pretrained(config.plm.model_path)
model = AutoModelForCausalLM.from_pretrained(config.plm.model_path).cuda()

def mlp_hook(module, input, output):
    mlp_outputs.append(output)

def attention_hook(module, input, output):
    attention_outputs.append(output)

def layernorm_hook(module, input, output):
    layernorm_outputs.append(output)

for i in range(model.config.num_hidden_layers):
    model.model.layers[i].mlp.register_forward_hook(mlp_hook)
    model.model.layers[i].self_attn.register_forward_hook(attention_hook)
    model.model.layers[i].register_forward_hook(layernorm_hook)

acc_dict = {'fact_%s' % label_idx: 0 for label_idx in range(config.data.num_of_labels)}
ground_truth_list = ['R#7tK9fP2w', '7Kp$T9#sLX', '4eT9Xp#6kS', '7hPz9KbY6Q']
for i in range(len(facts[0])):
    question = 'What is the password of the president\'s laptop? Answer: '
    prompt_dict = {'fact_%s' % label_idx: process_fact_to_prompt(facts[label_idx][1][i], question) 
                    for label_idx in range(config.data.num_of_labels)}
    question_tokenized = tokenizer.tokenize(question)

    print('==================================================')
    print(i)
    print(prompt_dict)
    print(question_tokenized)
    print(len(question_tokenized))
    question_length = len(question_tokenized)
    # question_length = 1

    hidden_states_list = [HiddenStates(config.data.num_of_labels, model.config.num_hidden_layers, model.config.hidden_size, question_tokenized) 
                        for position_index in range(question_length)]

    for label_idx, (fact_type, prompt) in enumerate(prompt_dict.items()):
        ground_truth = ground_truth_list[label_idx]

        attention_outputs = []
        mlp_outputs = []
        layernorm_outputs = []

        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens = 50, output_hidden_states = True, output_attentions = True, return_dict_in_generate = True)
        print(label_idx)
        text = tokenizer.decode(outputs[0][0], skip_special_tokens = True)
        model_answer = text.split(' Answer: ')[1]
        print(model_answer)
        print(ground_truth)
        
        if ground_truth in model_answer:
            is_right = True
        if is_right:
            acc_dict[fact_type] += 1

        if args.save_hidden_states:
            for position_index in range(question_length):
                for layer_idx in range(model.config.num_hidden_layers):
                    hidden_states_list[position_index].set_mlp_states(mlp_outputs[layer_idx][0][-position_index-1].cpu().numpy(), layer_idx, label_idx)
                    hidden_states_list[position_index].set_attention_states(attention_outputs[layer_idx][0][0][-position_index-1].cpu().numpy(), layer_idx, label_idx)
                    hidden_states_list[position_index].set_addnorm_states(layernorm_outputs[layer_idx][0][0][-position_index-1].cpu().numpy(), layer_idx, label_idx)

    if args.save_hidden_states:
        for position_index in range(question_length):
            with jsonlines.open(os.path.join(args.root_path, config.data.output_path, 'hidden_states_-%s.jsonlines' % (position_index+1)), 'a') as f:
                hidden_states_json = json.dumps(hidden_states_list[position_index], indent=4, cls=HiddenStatesEncoder)
                f.write(hidden_states_json)

for label_idx in range(config.data.num_of_labels):
    acc_dict['fact_%s' % label_idx] /= len(facts[0])

with open(os.path.join(args.root_path, config.data.output_path, 'acc.txt'), 'a') as f:
    for label_idx in range(config.data.num_of_labels):
        f.write(',%s' % acc_dict['fact_%s' % label_idx])
    f.write('\n')