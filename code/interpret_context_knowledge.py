from process_prompt import process_fact_to_prompt
from hidden_states_obj import HiddenStates
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
import os
import shutil
import torch
import argparse
import yaml
import json
import random
from datasets import load_dataset
from probing_multiple_steps_with_LR import ProbingMultipleSteps

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

parser = argparse.ArgumentParser()
parser.add_argument('--config_yaml', type=str)
parser.add_argument('--fact_idx', type=int, default=0, help='The fact ids involved, -1 for all fact ids.')
parser.add_argument('--root_path', required=True, type=str, help='Project root directory.')
parser.add_argument('--is_probing', type=bool, default=False, help='Whether to perform a probing task.')
parser.add_argument('--is_record_acc', type=bool, default=False, help='Whether the accuracy of the pilot experiment is recorded.')
parser.add_argument('--is_plot_heatmap', type=bool, default=False, help='Whether a heat map is required, for the case where fact_idx!=-1.')
parser.add_argument('--is_record_last_vi', type=bool, default=False, help='Whether to record the vi of the last token for plotting line graphs.')
parser.add_argument('--is_record_all_vi', type=bool, default=False, help='Whether to record the vi of all tokens for comparing entity tokens with non-entity tokens')
parser.add_argument('--num_of_irrelevant_evidence', type=int, default=0, help='Limited to the password task, detects the effect of the amount of irrelevant evidence on vi')
args = parser.parse_args()
with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

if config.data.task == 'commonsense':
    # dataset = load_dataset("osunlp/ConflictQA",'ConflictQA-popQA-gpt4')
    dataset = load_dataset(os.path.join(args.root_path, config.data.input_path, 'ConflictQA'),'ConflictQA-popQA-gpt4')
    dataset = dataset.sort('popularity', reverse = True)
    if args.num_of_irrelevant_evidence:
        raise ValueError('num_of_irrelevant_evidence must be set to 0 when config.data.task == \'commonsense\'!')

# Simply change to deepspeed for multi-GPUs
os.environ['CUDA_VISIBLE_DEVICES']=str(config.environment.cuda_visible_devices[0])

tokenizer = AutoTokenizer.from_pretrained(config.plm.model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(config.plm.model_path, torch_dtype=torch.float16).cuda()

def mlp_hook(module, input, output):
    mlp_outputs.append(output)

def attention_hook(module, input, output):
    attention_outputs.append(output)

def layer_outputs_hook(module, input, output):
    layer_outputs_outputs.append(output)

for i in range(model.config.num_hidden_layers):
    if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
        model.model.layers[i].mlp.register_forward_hook(mlp_hook)
        model.model.layers[i].self_attn.register_forward_hook(attention_hook)
    model.model.layers[i].register_forward_hook(layer_outputs_hook)

fact = pd.read_csv(os.path.join(args.root_path, config.data.input_path, '%s_evidence/fact_0.txt' % config.data.task), sep = '------', header = None, engine = 'python')
if args.fact_idx == -1:
    fact_idx_list = [_ for _ in range(max(fact[0])+1)]
else:
    fact_idx_list = [args.fact_idx]

for fact_idx in fact_idx_list:
    tensor_root_path = os.path.join(args.root_path, config.data.output_path, 'tensors_2')
    if not os.path.exists(tensor_root_path):
        os.makedirs(tensor_root_path)
    tensor_dirs = os.listdir(tensor_root_path)
    for tensor_dir in tensor_dirs:
        shutil.rmtree(os.path.join(tensor_root_path, tensor_dir), ignore_errors=True)

    facts = []
    for label_idx in range(config.data.num_of_labels):
        fact = pd.read_csv(os.path.join(args.root_path, config.data.input_path, '%s_evidence/fact_%s.txt' % (config.data.task, label_idx)), sep = '------', header = None, engine = 'python')
        fact = fact[fact[0] == fact_idx]
        fact = fact.reset_index(drop=True)
        facts.append(fact)

    acc_dict = {'fact_%s' % label_idx: 0 for label_idx in range(config.data.num_of_labels)}
    for i in range(len(facts[0])):
        if config.data.task == 'commonsense':
            question = dataset['test']['question'][facts[0][1][0]]
            ground_truth_list = [dataset['test']['ground_truth'][fact_idx]] * config.data.num_of_labels
            prompt_dict = {'fact_%s' % label_idx: process_fact_to_prompt(facts[label_idx][4][i], question) 
                            for label_idx in range(config.data.num_of_labels)}
        elif config.data.task == 'password':
            question = 'What is the password of the president\'s laptop?'
            ground_truth_list = [['R#7tK9fP2w'], ['7Kp$T9#sLX'], ['4eT9Xp#6kS'], ['7hPz9KbY6Q']]
            if args.num_of_irrelevant_evidence == 0:
                prompt_dict = {'fact_%s' % label_idx: process_fact_to_prompt(facts[label_idx][1][i], question) 
                                for label_idx in range(config.data.num_of_labels)}
            else:
                other_facts = pd.read_csv(os.path.join(args.root_path, config.data.input_path, 'commonsense_evidence/fact_0.txt'), sep = '------', header = None, engine = 'python')
                irrelevant_facts = other_facts.sample(args.num_of_irrelevant_evidence)
                final_fact =  facts[label_idx][1][i]
                for irrelevant_fact in irrelevant_facts[4]:
                    final_fact += '\n' + irrelevant_fact
                    prompt_dict = {'fact_%s' % label_idx: process_fact_to_prompt(final_fact, question) 
                                    for label_idx in range(config.data.num_of_labels)}

        question_tokenized = tokenizer.tokenize(question)

        print('==================================================')
        print(i)
        print(prompt_dict)
        print(question_tokenized)
        print(len(question_tokenized))
        if args.is_record_last_vi:
            question_tokenized = [question_tokenized[-1]]
        if args.is_record_all_vi:
            entity_tokenize = tokenizer.tokenize(facts[label_idx][2][i]) + tokenizer.tokenize(facts[label_idx][3][i])
            entity_tag_list = []
            for token in question_tokenized:
                if token in entity_tokenize:
                    entity_tag_list.append(1)
                else:
                    entity_tag_list.append(0)

        question_length = len(question_tokenized)

        if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
            is_record_attention_and_mlp = True
        else:
            is_record_attention_and_mlp = False
        hidden_states_list = [HiddenStates(config.data.num_of_labels, model.config.num_hidden_layers, model.config.hidden_size, question_tokenized, tensor_root_path, is_record_attention_and_mlp) 
                            for step_index in range(question_length)]

        for label_idx, (fact_type, prompt) in enumerate(prompt_dict.items()):
            ground_truth = ground_truth_list[label_idx]

            attention_outputs = []
            mlp_outputs = []
            layer_outputs_outputs = []

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens = 10, output_hidden_states = True, output_attentions = True, return_dict_in_generate = True)
            print(label_idx)
            text = tokenizer.decode(outputs[0][0], skip_special_tokens = True)
            # model_answer = text.split(' Answer: ')[1]
            model_answer = text
            print(model_answer)
            print(ground_truth)
            
            is_right = False
            for ground_truth_answer in ground_truth:
                if ground_truth_answer in model_answer:
                    is_right = True
            if is_right:
                acc_dict[fact_type] += 1

            if args.is_probing:
                for step_index in range(question_length):
                    for layer_idx in range(model.config.num_hidden_layers):
                        if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
                            hidden_states_list[step_index].set_mlp_states(torch.Tensor(mlp_outputs[layer_idx][0][-step_index-1]), layer_idx, label_idx)
                            hidden_states_list[step_index].set_attention_states(torch.Tensor(attention_outputs[layer_idx][0][0][-step_index-1]), layer_idx, label_idx)
                        hidden_states_list[step_index].set_layer_outputs(torch.Tensor(layer_outputs_outputs[layer_idx][0][0][-step_index-1]), layer_idx, label_idx)

        if args.is_probing:
            for step_index in range(question_length):
                if not os.path.exists(os.path.join(tensor_root_path, 'step_%s') % step_index):
                    os.makedirs(os.path.join(tensor_root_path, 'step_%s') % step_index)
                hidden_states_list[step_index].save_tensors(step_index)

    for label_idx in range(config.data.num_of_labels):
        acc_dict['fact_%s' % label_idx] /= len(facts[0])

    if args.is_record_acc:
        with open(os.path.join(args.root_path, config.data.output_path, 'acc_%s.txt' % config.data.task), 'a') as f:
            f.write(str(fact_idx))
            for label_idx in range(config.data.num_of_labels):
                f.write(',%s' % acc_dict['fact_%s' % label_idx])
            f.write('\n')

    if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
        is_record_attention_and_mlp = True
    else:
        is_record_attention_and_mlp = False
    if args.is_probing:
        probing_class = ProbingMultipleSteps(config, args, tensor_root_path, question_tokenized, len(facts[0]), is_record_attention_and_mlp)
        probing_class.probing()
        if args.is_plot_heatmap:
            probing_class.plot_heatmap()
        if args.is_record_last_vi:
            probing_class.record_last_vi()
        if args.is_record_all_vi:
            probing_class.record_all_vi(entity_tag_list)