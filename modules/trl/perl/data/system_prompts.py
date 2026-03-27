SYSTEM_PROMPT = """
You are a helpful AI Assistant that provides well-reasoned and detailed responses.
You first think about the reasoning process as an internal monologue and then provide the user with the answer.
Respond in the following format: <think>\n...\n</think>\n, then answer.
"""

def make_conversation(example):
    # check if key "problem" exists
    if "problem" in example:
        problem = example["problem"]
        answer = example["answer"]
    else:
        problem = example["prompt"] # open-r1/DAPO-Math-17k-Processed
        answer = example["solution"]

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT}, 
        {"role": "user", "content": problem}
    ]
    return {"prompt": prompt, "solution": answer}