from hidden_states_obj import HiddenStates, HiddenStatesEncoder
import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import copy
import argparse
import yaml
import json
import jsonlines
import random
import scipy

from probing_model import LinearClassifier, MLPClassifier, train
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
plt.rc('font', **font)
sns.set_style('darkgrid')

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

parser = argparse.ArgumentParser()
parser.add_argument('--config_yaml', type=str)
parser.add_argument('--fact_idx', type=int, default=0)
parser.add_argument('--root_path', type=str, default='/home/jtj/probing_llama')
args = parser.parse_args()
with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

with jsonlines.open(os.path.join(args.root_path, config.data.json_data_path, 'hidden_states_-1.jsonlines')) as f: 
    for line in f:
        hidden_states_json = json.loads(line)
        hidden_step = len(hidden_states_json['question_tokenized'])
        question_tokenized = hidden_states_json['question_tokenized']
        break

os.environ['CUDA_VISIBLE_DEVICES']=str(config.environment.cuda_visible_devices[0])
acc_of_each_position = {
    'mlp_states': [[] for _ in range(hidden_step)], 
    'attention_states': [[] for _ in range(hidden_step)], 
    'addnorm_states': [[] for _ in range(hidden_step)]
    }

for position_index in range(hidden_step):
    hidden_states_of_each_layer = {
        'mlp_states': [[] for _ in range(config.task.layer_num)],
        'attention_states': [[] for _ in range(config.task.layer_num)],
        'addnorm_states': [[] for _ in range(config.task.layer_num)]
        }
    labels_of_each_layer = {
        'mlp_states': [[] for _ in range(config.task.layer_num)],
        'attention_states': [[] for _ in range(config.task.layer_num)],
        'addnorm_states': [[] for _ in range(config.task.layer_num)]
        }
    with jsonlines.open(os.path.join(args.root_path, config.data.json_data_path, 'hidden_states_-%s.jsonlines' % (position_index+1))) as f: 
        for line in f:
            hidden_states_json = json.loads(line)
            for position in acc_of_each_position.keys():
                for layer_idx in range(config.task.layer_num):
                    for label_idx in range(config.data.num_of_labels):
                        state = hidden_states_json[position][label_idx][layer_idx]
                        hidden_states_of_each_layer[position][layer_idx].append(state)
                        labels_of_each_layer[position][layer_idx].append(label_idx)
    
    for layer_idx in range(config.task.layer_num):
        for position, acc_list in acc_of_each_position.items():
            X_train, X_test, y_train, y_test = train_test_split(torch.Tensor(hidden_states_of_each_layer[position][layer_idx]).cuda(), torch.Tensor(labels_of_each_layer[position][layer_idx]).cuda(), test_size = 0.2)
            y_train = F.one_hot(y_train.long(), num_classes = config.data.num_of_labels).float()
            y_test = y_test.long()

            H_yb = 0
            H_yx = 0

            X_train_null = torch.zeros_like(X_train)
            # for i in range(len(X_train_null)):
            #     X_train_null[i] = torch.Tensor(no_state)

            model = LinearClassifier(X_train.shape[-1], config.data.num_of_labels).cuda()
            # model = MLPClassifier(X_train.shape[-1], config.data.num_of_labels).cuda()
            model = train(model, X_train_null, y_train)
            with torch.no_grad():
                y_pred = model(X_test)

            H_yb = 0
            for i, ground_truth in enumerate(y_test):
                H_yb += -1 * torch.log2(y_pred[i][ground_truth]).item()
            H_yb /= len(y_pred)

            # loss_fun = torch.nn.CrossEntropyLoss()
            # H_yb = loss_fun(y_pred, y_test).item()


            model = LinearClassifier(X_train.shape[-1], config.data.num_of_labels).cuda()
            # model = MLPClassifier(X_train.shape[-1], config.data.num_of_labels).cuda()
            model = train(model, X_train, y_train)
            with torch.no_grad():
                y_pred = model(X_test)

            H_yx = 0
            for i, ground_truth in enumerate(y_test):
                H_yx += -1 * torch.log2(y_pred[i][ground_truth]).item()
            H_yx /= len(y_pred)

            # loss_fun = torch.nn.CrossEntropyLoss()
            # H_yx = loss_fun(y_pred, y_test).item()

            # for i in range(len(y_pred)):
            #     prob = y_pred[i][y_test[i]]
            #     H_yx += -1 * torch.log2(prob)
            # H_yx /= len(y_pred)
            # H_yx = H_yx.item()
            
            Vi = H_yb - H_yx

            # y_pred = lr.predict(X_test)
            # test_f1 = f1_score(y_test, y_pred)
            print('==============================================')
            print('position: %s, position idx: %s, layer idx: %s' % (position, -(position_index+1), layer_idx))
            print('Hyb: %s, Hyx: %s, pvi: %s' % (H_yb, H_yx, Vi))
            # print('Vi: %s' % Vi)
            acc_list[position_index].append(Vi)

for position, acc_list in acc_of_each_position.items():
    with open(os.path.join(args.root_path, config.data.json_data_path, 'results.txt'), 'a') as f:
        f.write(position)
        f.write(',')
        f.write(str(args.fact_idx))
        for vi in acc_list[0]:
            f.write(',%s' % vi)
        f.write('\n')

for position in acc_of_each_position.keys():
    acc_of_each_position[position] = np.array(acc_of_each_position[position])

# for position_index in range(hidden_step):
#     mlp_states_of_word_answer = acc_of_each_position['mlp_states'][position_index]
#     attention_states_of_word_answer = acc_of_each_position['attention_states'][position_index]
#     addnorm_states_of_word_answer = acc_of_each_position['addnorm_states'][position_index]
#     layer_np = np.array([i for i in range(config.task.layer_num)])
#     plt.plot(layer_np, mlp_states_of_word_answer, label = 'MLP')
#     plt.plot(layer_np, attention_states_of_word_answer, label = 'Attention')
#     plt.plot(layer_np, addnorm_states_of_word_answer, label = 'LayerNorm')
#     plt.legend()
#     plt.xlabel('Layer')
#     plt.ylabel('Vi')
#     plt.savefig(os.path.join(args.root_path, config.data.json_data_path, 'vi_of_word_index_%s.jpg' % -(position_index+1)))
#     plt.close()

vmin = min(v.min() for v in acc_of_each_position.values())
vmax = min(v.max() for v in acc_of_each_position.values())
fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[config.task.layer_num, config.task.layer_num, config.task.layer_num, 1]), 
                        figsize=(config.task.layer_num*3+1, hidden_step))
ax1 = sns.heatmap(np.flip(acc_of_each_position['mlp_states'], axis=0), yticklabels=question_tokenized, cbar=False, ax = axs[0], vmin=vmin, vmax=vmax)
ax2 = sns.heatmap(np.flip(acc_of_each_position['attention_states'], axis=0), yticklabels=question_tokenized, cbar=False, ax = axs[1], vmin=vmin, vmax=vmax)
ax3 = sns.heatmap(np.flip(acc_of_each_position['addnorm_states'], axis=0), yticklabels=question_tokenized, cbar=False, ax = axs[2], vmin=vmin, vmax=vmax)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Step')
ax2.set_xlabel('Layer')
ax3.set_xlabel('Layer')
fig.colorbar(axs[1].collections[0], cax=axs[3])
# figure = ax.get_figure()
plt.savefig(os.path.join(args.root_path, config.data.json_data_path, 'vi_heatmap.pdf'), bbox_inches='tight')
plt.close()