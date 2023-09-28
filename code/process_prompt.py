import pandas as pd

def process_fact_to_prompt(fact, question):
    prompt1 = 'Here are some confirmed evidence, don\'t go doubting it.\n'
    prompt2 = 'Please answer the question based solely on the evidence above. '
    # prompt1 = ''
    # prompt2 = ''
    # prompt1 = 'Here are some fake evidence, you don\'t need to focus much on it.\n'
    # prompt2 = 'Please answer the question based solely on the true facts, not on the fake evidence provided above. '

    return prompt1 + fact + '\n' + prompt2 + question
    # return prompt1 + '\n' + fact + '\n' + question + '\n Answer: '
    # return fact + '\n' + question + '\n Answer: '