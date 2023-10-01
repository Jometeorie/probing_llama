import torch
import os

class HiddenStates:
    def __init__(self, num_of_labels, layer_num, dimension, question_tokenized, tensor_root_path):
        self.mlp_states = torch.zeros((num_of_labels, layer_num, dimension))
        self.attention_states = torch.zeros((num_of_labels, layer_num, dimension))
        self.layer_outputs = torch.zeros((num_of_labels, layer_num, dimension))
        self.question_tokenized = question_tokenized
        self.num_of_labels = num_of_labels
        self.layer_num = layer_num
        self.tensor_root_path = tensor_root_path
    
    def set_mlp_states(self, mlp_states_per_layer, layer_idx, label_idx):
        self.mlp_states[label_idx][layer_idx] = mlp_states_per_layer

    def set_attention_states(self, attention_states_per_layer, layer_idx, label_idx):
        self.attention_states[label_idx][layer_idx] = attention_states_per_layer
    
    def set_layer_outputs(self, layer_outputs_per_layer, layer_idx, label_idx):
        self.layer_outputs[label_idx][layer_idx] = layer_outputs_per_layer
    
    def save_tensors(self, step_index):
        self.save_file(self.mlp_states, 'mlp_states', step_index)
        self.save_file(self.attention_states, 'attention_states', step_index)
        self.save_file(self.layer_outputs, 'layer_outputs', step_index)
    
    def save_file(self, states, position, step_index):
        for layer_idx in range(self.layer_num):
            tensor_save_path = os.path.join(self.tensor_root_path, 'step_%s' % step_index, '%s_layer_%s.pt' % (position, layer_idx))
            if not os.path.exists(tensor_save_path):
                torch.save(states[:, layer_idx], tensor_save_path)
            else:
                pre_states = torch.load(tensor_save_path)
                torch.save(torch.cat((pre_states, states[:, layer_idx]), 0), tensor_save_path)