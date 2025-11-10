
import torch
import os
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from open_tinker.utils.utils import (
    parse_toml_to_args, # toml to args
    make_conversation # add system prompt to dataset
)
from open_tinker.utils.logging import init_logger, logger
from open_tinker.trl_trainer.grpo import format_reward, accuracy_reward

def fuzzy_jobs(
    config_path: str,
    output_dir: str
):
    args = parse_toml_to_args(config_path)
    init_logger()
    args.training.output_dir = output_dir
    if not os.path.exists(output_dir): # check if output_dir exists
        os.makedirs(output_dir)
    else:
        logger.info(f"Output directory {output_dir} already exists, using it")
    set_seed(args.training.seed)
    
    return args

def train(
    config_path: str,
    output_dir: str
):
    # 0. parse args and prepare logger
    args = fuzzy_jobs(config_path, output_dir)

    # 1. load dataset
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    train_dataset, test_dataset = load_dataset(args.dataset.dataset_name_or_path, split=['train[:5%]', 'test[:5%]'])
    train_dataset = train_dataset.map(make_conversation).remove_columns(['messages', 'problem'])
    test_dataset = test_dataset.map(make_conversation)
    logger.info(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples")

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path, 
        torch_dtype= torch.bfloat16 if args.training.dtype == "bfloat16" else torch.float16, 
        device_map="auto"
    )
    logger.info(f"Model loaded successfully")

    # 3. configure lora
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, configuring lora")
        lora_config = LoraConfig(
            task_type=args.peft.task_type,
            use_dora=args.peft.use_dora,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            lora_dropout=args.peft.lora_dropout,
            target_modules=args.peft.target_modules,
        )
        model = get_peft_model(model, lora_config)
        logger.info(f"Lora configured successfully")
    
    model.print_trainable_parameters()

    # 4.Load and configure tokenizer for left padding
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if hasattr(model.config, 'pad_token_id') and model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


    # 5.Training configuration
    training_args = GRPOConfig(
        output_dir=args.training.output_dir,
        learning_rate=args.training.lr,
        remove_unused_columns=args.training.remove_unused_columns,
        gradient_accumulation_steps=args.training.gradient_accumulation_steps,
        num_train_epochs=args.training.num_train_epochs,
        bf16=True if args.training.dtype == "bfloat16" else False,
        max_completion_length=args.training.max_completion_length,
        num_generations=args.training.num_generations,
        max_prompt_length=args.training.max_prompt_length,
        logging_steps=args.training.logging_steps,
        save_strategy=args.training.save_strategy,
        save_steps=args.training.save_steps,
        use_vllm=args.training.use_vllm,
    )

    # 6.Train
    logger.info(f"Training model with GRPO")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, accuracy_reward],
        args=training_args,
        train_dataset=train_dataset
    )
    trainer.train()
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    Fire(train)