import pandas as pd

def llama_v2_prompt(messages):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = f"""You are a helpful, respectful and honest assistant."""
    if messages[0]["role"] != "system":
        messages = [
            {
                "role": "system",
                "content": DEFAULT_SYSTEM_PROMPT,
            }
        ] + messages
    messages = [
        {
            "role": messages[1]["role"],
            "content": B_SYS + messages[0]["content"] + E_SYS + messages[1]["content"],
        }
    ] + messages[2:]
    messages_list = [
        f"{BOS}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {EOS}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]
    messages_list.append(f"{BOS}{B_INST} {(messages[-1]['content']).strip()} {E_INST}")
    return "".join(messages_list)


def process_fact_to_prompt(fact, question):
    prompt1 = 'Here are some confirmed facts, don\'t go doubting it.\n'
    prompt2 = 'Please answer the question based solely on the evidence above. Please directly answer a single entity. '
    prompt = prompt1 + fact + '\n' + prompt2 + question
    prompt = llama_v2_prompt([{'role': 'user', 'content': prompt}])
    # prompt += ' The password is:'
    return prompt