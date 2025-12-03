from typing import Any, Dict, Iterable, List, Tuple
from datasets import load_dataset

from utils import extract_boxed_answer



def load_dataset_from_hf():
    return load_dataset('HuggingFaceH4/aime_2024', split='train')



def prepare_prompt(sample) -> str:
    return '''Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

{problem}
'''.format(problem=sample['problem'])




def score_response(prompt: str, response: str, sample: Dict[str, Any]) -> Tuple[float, Dict]:
    ground_truth = int(sample['answer'])
    solution = int(extract_boxed_answer(response))
    
    # 比较答案是否正确，返回分数（1.0表示正确，0.0表示错误）
    score = 1.0 if ground_truth == solution else 0.0

    return score, {'ground_truth': ground_truth, 'solution': solution}

