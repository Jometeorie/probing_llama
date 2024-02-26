# probing_llama
## Experimental Setup
Access the [scipts/](https://github.com/Jometeorie/probing_llama/blob/master/scripts/) directory and excute the corresponding script. Ensure modification of the paths for storing the model and the project root directory, specifically adjusting the 'model_path' in the YAML configuration and the 'root_path' in each script accordingly.

## Preliminary Experiments
Preliminary experiments were conducted to assess: 1. accuracy in responding to questions with factual and counterfactual evidence; 2. accuracy in responding to questions with varied password inputs.

The scripts for both of these components can be found in [scipts/run_preliminary_experiments.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_preliminary_experiments.sh). The final results are reflected in the files 'acc_commonsense.txt' and 'acc_password.txt' located within the respective folders for different models under [results/](https://github.com/Jometeorie/probing_llama/blob/master/results).

## Testing of Capacities for Conflict Knowledge
### Model Processing of vi Heatmaps for Individual Facts

The scripts for this part can be found in [scripts/run_heatmap_for_commonsense.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_heatmap_for_commonsense.sh). The resulting file is vi_heatmap.pdf in the different models folder in [results/](https://github.com/Jometeorie/probing_llama/blob/master/results).

### Line Graph of the Average vi of for Processing the Last Token

The scripts for this part can be found in [scripts/record_vi_of_last_token.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/record_vi_of_last_token.sh). The resulting file is vi.txt in the different models folder in [results/](https://github.com/Jometeorie/probing_llama/blob/master/results).

### Testing for vi Differences between Entity and Non-Entity tokens
The scripts for this part can be found in [scripts/record_vi_of_all_tokens.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/record_vi_of_all_tokens.sh)，The resulting file is all_vi.txt in the different models folder in [results/](https://github.com/Jometeorie/probing_llama/blob/master/results).

## Testing of Capacities for Newly Acquired Knowledge
### Model Processing of vi Heatmaps for Newly Acquired Knowledge
The scripts for this part can be found in [scripts/run_heatmap_for_password.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/run_heatmap_for_password.sh).

### Long-Term Memory Capability
The scripts for this part can be found in [scripts/record_vi_with_irrelevant_evidence.sh](https://github.com/Jometeorie/probing_llama/blob/master/scripts/record_vi_with_irrelevant_evidence.sh). The resulting file is password_last_vi_irr_{0-10}.txt in the different models folder in [results/](https://github.com/Jometeorie/probing_llama/blob/master/results).