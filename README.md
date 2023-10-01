# probing_llama
## 实验设置
1. 运行期

## 先导实验
先导实验测试1. 测试llama接受factual evidence和counterfactual后回答问题的准确率；2. 测试llama接受不同密码后回答准确率。

这两部分的脚本均在[scipts/run_preliminary_experiments.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_preliminary_experiments.sh)中，其中关于非chat模型的实验可选，这里仅想证明经过chat微调后的llama在回答事实问题时具有更高准确率，因此接下来的工作主要围绕chat模型展开。

最终得到的文件为[results/](https://github.com/Jometeorie/probing_llama/blob/master/results)中不同模型文件夹下的acc_commonsense.txt和acc_password.txt。

## 冲突知识能力检测
### 测试模型处理单个fact的vi热力图

该部分的脚本为[scripts/run_heatmap_for_commonsense.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_heatmap_for_commonsense.sh)，仅需选择一个看起来比较好看的fact作为展示图即可。可现在llama-7b中测试，最后尽量换成llama-70b作为论文展示图。

最终得到的文件为[results/](https://github.com/Jometeorie/probing_llama/blob/master/results)不同模型文件夹下的vi_heatmap.pdf。

### 测试模型处理不同facts最后一个token的平均vi折线图

该部分的脚本为[scripts/record_vi_of_last_token.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/record_vi_of_last_token.sh)，最终得到的文件为[results/](https://github.com/Jometeorie/probing_llama/blob/master/results)不同模型文件夹下的vi.txt，用于后续绘图。

### 测试实体词和非实体词间的vi差异
该部分的脚本为[scripts/record_vi_of_all_tokens.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/record_vi_of_all_tokens.sh)，最终得到的文件为[results/](https://github.com/Jometeorie/probing_llama/blob/master/results)不同模型文件夹下的all_vi.txt，用于后续绘图。

## 非冲突全新知识检测
### 测试模型处理知识的vi热力图
同上节，执行脚本为[scripts/run_heatmap_for_commonsense.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_heatmap_for_commonsense.sh)。

### 长时间记忆能力
待完成。。。