from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse, verify

from .system_prompts import SYSTEM_PROMPT
import re

# reward function from https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py
def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if re.match(pattern, content) else 0.0 for content in completion_contents]


def accuracy_reward(completions, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    solutions = kwargs['solution']
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, solution in zip(completion_contents, solutions):
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        answer_parsed = parse(content, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) != 0:
            try:
                rewards.append(float(verify(answer_parsed, gold_parsed)))
            except Exception:
                rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards

def load_tinyzero_dataset(dataset_name_or_path: str, example_numbers: int = None):

    train_dataset = load_dataset(dataset_name_or_path, split="train")
        
    if example_numbers is not None and len(train_dataset) > example_numbers:
        train_dataset = train_dataset.select(range(example_numbers))
    
    return {
        "train_dataset": train_dataset,
        "test_dataset": None,
        "reward_functions": [format_reward, accuracy_reward],
    }

