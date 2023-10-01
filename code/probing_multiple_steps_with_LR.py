from hidden_states_obj import HiddenStates
import numpy as np
import os
import torch
import torch.nn.functional as F

from probing_model import LinearClassifier, MLPClassifier, train
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns


class ProbingMultipleSteps:
    def __init__(self, config, args, tensor_root_path, question_tokenized, fact_num):
        self.config = config
        self.args = args
        self.tensor_root_path = tensor_root_path
        self.question_tokenized = question_tokenized
        self.fact_num = fact_num

        self.acc_of_each_position = {
            'mlp_states': [[] for _ in range(len(question_tokenized))], 
            'attention_states': [[] for _ in range(len(question_tokenized))], 
            'layer_outputs': [[] for _ in range(len(question_tokenized))]
        }
        
        self.set_plot_config()
    
    def probing(self):
        for step_index in range(len(self.question_tokenized)):
            labels = [_ for _ in range(self.config.data.num_of_labels)] * self.fact_num

            for layer_idx in range(self.config.plm.layer_num):
                for position, acc_list in self.acc_of_each_position.items():
                    states = torch.load(os.path.join(self.tensor_root_path, 'step_%s' % step_index, '%s_layer_%s.pt' % (position, layer_idx)))
                    X_train, X_test, y_train, y_test = train_test_split(states.cuda(), torch.Tensor(labels).cuda(), test_size = 0.2)
                    y_train = F.one_hot(y_train.long(), num_classes = self.config.data.num_of_labels).float()
                    y_test = y_test.long()

                    H_yb = 0
                    H_yx = 0

                    X_train_null = torch.zeros_like(X_train)
                    # for i in range(len(X_train_null)):
                    #     X_train_null[i] = torch.Tensor(no_state)

                    model = LinearClassifier(X_train.shape[-1], self.config.data.num_of_labels).cuda()
                    # model = MLPClassifier(X_train.shape[-1], self.config.data.num_of_labels).cuda()
                    model = train(model, X_train_null, y_train)
                    with torch.no_grad():
                        y_pred = model(X_test)

                    H_yb = 0
                    for i, ground_truth in enumerate(y_test):
                        H_yb += -1 * torch.log2(y_pred[i][ground_truth]).item()
                    H_yb /= len(y_pred)

                    # loss_fun = torch.nn.CrossEntropyLoss()
                    # H_yb = loss_fun(y_pred, y_test).item()


                    model = LinearClassifier(X_train.shape[-1], self.config.data.num_of_labels).cuda()
                    # model = MLPClassifier(X_train.shape[-1], self.config.data.num_of_labels).cuda()
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
                    print('position: %s, position idx: %s, layer idx: %s' % (position, -(step_index+1), layer_idx))
                    print('Hyb: %s, Hyx: %s, pvi: %s' % (H_yb, H_yx, Vi))
                    # print('Vi: %s' % Vi)
                    acc_list[step_index].append(Vi)
    
    def record_last_vi(self):
        for position, acc_list in self.acc_of_each_position.items():
            with open(os.path.join(self.args.root_path, self.config.data.output_path, 'last_vi.txt'), 'a') as f:
                f.write(position)
                for vi in acc_list[0]:
                    f.write(',%s' % vi)
                f.write('\n')
    
    def record_all_vi(self, entity_tag_list):
        for position, acc_list in self.acc_of_each_position.items():
            with open(os.path.join(self.args.root_path, self.config.data.output_path, 'all_vi.txt'), 'a') as f:
                acc_np = np.flip(np.array(acc_list), axis=0)
                for i in range(len(acc_np)):
                    f.write('%s---%s---%s' % (self.question_tokenized[i], entity_tag_list[i], position))
                    for vi in acc_np[i]:
                        f.write('---%s' % vi)
                    f.write('\n')

    def set_plot_config(self):
        self.font = {
            'family' : 'Times New Roman',
            'weight' : 'normal',
            'size'   : 14,
        }
        plt.rc('font', **self.font)
        sns.set_style('darkgrid')

    def plot_heatmap(self):
        for position in self.acc_of_each_position.keys():
            self.acc_of_each_position[position] = np.array(self.acc_of_each_position[position])

        vmin = min(v.min() for v in self.acc_of_each_position.values())
        vmax = min(v.max() for v in self.acc_of_each_position.values())
        fig, axs = plt.subplots(ncols=4, gridspec_kw=dict(width_ratios=[self.config.plm.layer_num, self.config.plm.layer_num, self.config.plm.layer_num, 1]), 
                                figsize=(self.config.plm.layer_num*3+1, len(self.question_tokenized)))
        ax1 = sns.heatmap(np.flip(self.acc_of_each_position['mlp_states'], axis=0), yticklabels=self.question_tokenized, cbar=False, ax = axs[0], vmin=vmin, vmax=vmax)
        ax2 = sns.heatmap(np.flip(self.acc_of_each_position['attention_states'], axis=0), yticklabels=self.question_tokenized, cbar=False, ax = axs[1], vmin=vmin, vmax=vmax)
        ax3 = sns.heatmap(np.flip(self.acc_of_each_position['layer_outputs'], axis=0), yticklabels=self.question_tokenized, cbar=False, ax = axs[2], vmin=vmin, vmax=vmax)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Step')
        ax2.set_xlabel('Layer')
        ax3.set_xlabel('Layer')
        fig.colorbar(axs[1].collections[0], cax=axs[3])
        # figure = ax.get_figure()
        plt.savefig(os.path.join(self.args.root_path, self.config.data.output_path, 'vi_heatmap.pdf'), bbox_inches='tight')
        plt.close()
            
    # def plot_line_graph(self):
    #     for position_index in range(len(question_tokenized)):
    #         mlp_states_of_word_answer = self.acc_of_each_position['mlp_states'][position_index]
    #         attention_states_of_word_answer = self.acc_of_each_position['attention_states'][position_index]
    #         layer_outputsof_word_answer = self.acc_of_each_position['layer_outputs'][position_index]
    #         layer_np = np.array([i for i in range(self.config.plm.layer_num)])
    #         plt.plot(layer_np, mlp_states_of_word_answer, label = 'MLP')
    #         plt.plot(layer_np, attention_states_of_word_answer, label = 'Attention')
    #         plt.plot(layer_np, layer_outputs_of_word_answer, label = 'LayerNorm')
    #         plt.legend()
    #         plt.xlabel('Layer')
    #         plt.ylabel('Vi')
    #         plt.savefig(os.path.join(self.args.root_path, self.config.data.output_path, 'vi_of_word_index_%s.pdf' % -(position_index+1)), bbox_inches='tight')
    #         plt.close()