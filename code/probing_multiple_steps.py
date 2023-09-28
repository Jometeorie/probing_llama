from hidden_states_obj import HiddenStates, HiddenStatesEncoder
import numpy as np
import pandas as pd
import os
import torch
import copy
import argparse
import yaml
import json
import jsonlines
import random
import scipy

import matplotlib.pyplot as plt
import seaborn as sns

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

parser = argparse.ArgumentParser()
parser.add_argument('--config_yaml', type=str)
args = parser.parse_args()
with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

hidden_step = 12

os.environ['CUDA_VISIBLE_DEVICES']=str(config.environment.cuda_visible_devices[0])
# acc_list = [[] for _ in range(hidden_step)]
true_false_cos_list = [[] for _ in range(hidden_step)]
true_no_cos_list = [[] for _ in range(hidden_step)]
false_no_cos_list = [[] for _ in range(hidden_step)]

for position_index in range(hidden_step):
    hidden_states_of_each_layer = [[] for _ in range(config.task.layer_num)]
    labels_of_each_layer = [[] for _ in range(config.task.layer_num)]
    with jsonlines.open(os.path.join(config.data.json_data_path, 'hidden_states_-%s.jsonlines' % (position_index+1))) as f:
        for line in f:
            hidden_states_json = json.loads(line)
            for layer_idx in range(config.task.layer_num):
                true_state = hidden_states_json[config.task.position + '_true'][layer_idx]
                false_state = hidden_states_json[config.task.position + '_false'][layer_idx]
                no_state = hidden_states_json[config.task.position + '_no'][layer_idx]
                hidden_states_of_each_layer[layer_idx].append(true_state)
                hidden_states_of_each_layer[layer_idx].append(false_state)
                hidden_states_of_each_layer[layer_idx].append(no_state)
                labels_of_each_layer[layer_idx].append(2)
                labels_of_each_layer[layer_idx].append(1)
                labels_of_each_layer[layer_idx].append(0)
    
    for layer_idx in range(config.task.layer_num):
        true_states = []
        false_states = []
        no_states = []
        for i in range(len(labels_of_each_layer[layer_idx])):
            if labels_of_each_layer[layer_idx][i] == 2:
                true_states.append(hidden_states_of_each_layer[layer_idx][i])
            elif labels_of_each_layer[layer_idx][i] == 1:
                false_states.append(hidden_states_of_each_layer[layer_idx][i])
            else:
                no_states.append(hidden_states_of_each_layer[layer_idx][i])
        true_false_cos = 0
        true_no_cos = 0
        false_no_cos = 0
        num_of_iter = 1000
        for _ in range(num_of_iter):
            true_state = random.choice(true_states)
            false_state = random.choice(false_states)
            no_state = random.choice(no_states)
            true_false_cos += 1 - scipy.spatial.distance.cosine(true_state, false_state)
            true_no_cos += 1 - scipy.spatial.distance.cosine(true_state, no_state)
            false_no_cos += 1 - scipy.spatial.distance.cosine(false_state, no_state)
        true_false_cos /= num_of_iter
        true_no_cos /= num_of_iter
        false_no_cos /= num_of_iter
        print('==============================================')
        print('position idx:', -(position_index+1))
        print('layer idx:', layer_idx)
        print('true false cos: %s, true no cos: %s, false no cos: %s' % (true_false_cos, true_no_cos, false_no_cos))
        true_false_cos_list[position_index].append(true_false_cos)
        true_no_cos_list[position_index].append(true_no_cos)
        false_no_cos_list[position_index].append(false_no_cos)

true_false_cos_np = np.array(true_false_cos_list)
true_no_cos_np = np.array(true_no_cos_list)
false_no_cos_np = np.array(false_no_cos_list)

# plt.figure(figsize=(config.task.layer_num*3+4, hidden_step))
vmin = min(true_false_cos_np.min(), true_no_cos_np.min(), false_no_cos_np.min())
vmax = max(true_false_cos_np.max(), true_no_cos_np.max(), false_no_cos_np.max())
fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[config.task.layer_num, config.task.layer_num, config.task.layer_num, 1]), 
                        figsize=(config.task.layer_num*3+1, hidden_step))
ax1 = sns.heatmap(true_false_cos_np, cbar=False, ax = axs[0], vmin=vmin, vmax=vmax)
ax2 = sns.heatmap(true_no_cos_np, yticklabels=False, cbar=False, ax = axs[1], vmin=vmin, vmax=vmax)
ax3 = sns.heatmap(false_no_cos_np, yticklabels=False, cbar=False, ax = axs[2], vmin=vmin, vmax=vmax)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Step')
ax2.set_xlabel('Layer')
ax3.set_xlabel('Layer')
fig.colorbar(axs[1].collections[0], cax=axs[3])
# figure = ax.get_figure()
plt.savefig(os.path.join(config.data.json_data_path, '%s_cos_heatmap.jpg' % config.task.position))
plt.close()