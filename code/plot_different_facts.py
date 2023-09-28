import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,
}
plt.rc('font', **font)
sns.set_style('darkgrid')

for model_name in ['Llama-2-7b-chat', 'Llama-2-13b-chat']:
    all_data = pd.read_csv('../results/%s-hf/results.txt' % model_name, header = None)
    if '13b' in model_name:
        layer_np = np.array([i for i in range(40)])
    else:
        layer_np = np.array([i for i in range(32)])
    for state in ['mlp_states', 'attention_states', 'addnorm_states']:
    # for state in ['addnorm_states']:
        data = all_data[all_data[0] == state]
        important_fact_data = data[data[1] < 50]
        important_fact_data = important_fact_data.drop([0, 1], axis = 1).to_numpy()
        # important_fact_data = sum(important_fact_data) / len(important_fact_data)
        avg = np.mean(important_fact_data, axis = 0)
        std = np.std(important_fact_data, axis=0)

        # r1 = list(map(lambda x: x[0]-x[1], zip(avg, std)))
        # r2 = list(map(lambda x: x[0]+x[1], zip(avg, std)))
        # plt.plot(layer_np, avg, label = '%s/%s' % (model_name, state))
        # plt.fill_between(layer_np, r1, r2, alpha=0.2)

        plt.plot(layer_np, avg, label = '%s/%s' % (model_name, state))

plt.legend()
plt.xlabel('Layer')
plt.ylabel('Vi')
plt.savefig('vi_of_different_facts.jpg')