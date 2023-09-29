import numpy as np
import json
from json import JSONEncoder

class HiddenStates:
    def __init__(self, num_of_labels, layer_num, dimension, question_tokenized):
        self.mlp_states = np.zeros((num_of_labels, layer_num, dimension))
        self.attention_states = np.zeros((num_of_labels, layer_num, dimension))
        self.addnorm_states = np.zeros((num_of_labels, layer_num, dimension))
        self.question_tokenized = question_tokenized
    
    def set_mlp_states(self, mlp_states_per_layer, layer_idx, label_idx):
        self.mlp_states[label_idx][layer_idx] = mlp_states_per_layer

    def set_attention_states(self, attention_states_per_layer, layer_idx, label_idx):
        self.attention_states[label_idx][layer_idx] = attention_states_per_layer
    
    def set_addnorm_states(self, addnorm_states_per_layer, layer_idx, label_idx):
        self.addnorm_states[label_idx][layer_idx] = addnorm_states_per_layer

class HiddenStatesEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj.__dict__
        # return JSONEncoder.default(self, obj)