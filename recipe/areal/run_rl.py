import sys

from areal import PPOTrainer
from areal.api.cli_args import GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.utils.hf_utils import load_hf_tokenizer

from datasets import load_dataset

AIME_TEMPLATE="""
Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{{$Answer}} where $Answer is the answer to the problem.\n\n{problem}\n\nRemember to put your answer on its own line after \"Answer:\".
"""

def get_dapo_math_17k(
    path: str,
):
    dataset = load_dataset(path=path, split="train")

    def process(sample):
        return {"messages": sample["prompt"], "answer": sample["label"]}

    dataset = dataset.map(process).remove_columns(["prompt", "label"])
    return dataset

def get_aime_2024(
    path: str,
):
    dataset = load_dataset(path=path, split="train")

    def process(sample):
        return {"messages": sample["prompt"], "answer": sample["label"]}

    dataset = dataset.map(process).remove_columns(["prompt", "label"])
    return dataset


def main(args):
    config, _ = load_expr_config(args, GRPOConfig)

    train_dataset = get_dapo_math_17k(config.train_dataset.path)
    valid_dataset = get_aime_2024(config.valid_dataset.path)

    workflow_kwargs = dict(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
        enable_thinking=False,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["gconfig"] = config.gconfig.new(temperature=0.6)

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="areal.workflow.rlvr.RLVRWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="areal.workflow.rlvr.RLVRWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
