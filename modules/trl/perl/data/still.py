"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from typing import Callable, Dict, Optional

from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        # Parse gold solution with default config (no extraction_config)
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, 0.0 otherwise to avoid NaN
            try:
                # Add timeout protection to avoid hanging
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("verify operation timed out")
                
                # Set 5 second timeout for verification
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                finally:
                    signal.alarm(0)  # Cancel the alarm
            except (Exception, TimeoutError) as e:
                print(f"verify failed: {e}, completion: {content}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = 0.0  # Return 0 instead of None to avoid NaN
        else:
            # If the gold solution is not parseable, assign 0 reward to avoid NaN
            reward = 0.0
            print(f"Failed to parse gold solution: {sol}")
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""

    def count_tags(text: str) -> float:
        count = 0.0
        # We only count </think> tag, because <think> tag is available in system prompt
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def prepare_still_example(example):
    """Convert STILL dataset format to the format expected by GRPO trainer.
    
    Input format:
    - prompt: already a conversation list (no conversion needed)
    - reward_model: {"ground_truth": "answer", "style": "rule"}
    
    Output format:
    - prompt: conversation list
    - solution: ground truth answer for reward computation
    """
    return {
        "prompt": example["prompt"],
        "solution": example["reward_model"]["ground_truth"]
    }


def load_still_dataset(dataset_name_or_path: str, example_numbers: int = None):
    """Load STILL dataset for GRPO training.
    
    Args:
        dataset_name_or_path: Path or name of the dataset
        example_numbers: Optional limit on number of examples to use
    
    Returns:
        Dictionary with train_dataset, test_dataset, reward_functions, and reward_weights
    """
    dataset = load_dataset(
        dataset_name_or_path, split="train"
    )

    # Convert dataset format: extract ground_truth from reward_model as solution
    dataset = dataset.map(prepare_still_example)

    if example_numbers is not None and len(dataset) > example_numbers:
        dataset = dataset.select(range(example_numbers))

    return {
        "train_dataset": dataset,
        "test_dataset": None,
        "reward_functions": [format_reward, accuracy_reward],
        "reward_weights": [1.0, 2.0]
    }