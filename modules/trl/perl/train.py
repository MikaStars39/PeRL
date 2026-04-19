
import torch
import os

from typing import List, Optional
from datasets import load_dataset
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from fire import Fire

from perl.utils.logging import init_logger, logger
from perl.data import load_dataset
from perl.config.config import TrainConfig

def fuzzy_jobs(
    args: TrainConfig
):
    init_logger()
    args.training.output_dir = args.training.output_dir or "output"
    args.training.run_name = args.training.run_name or args.training.output_dir # training run name is the output_dir
    if not os.path.exists(args.training.output_dir): # check if output_dir exists
        os.makedirs(args.training.output_dir, exist_ok=True)
    else:
        logger.info(f"Output directory {args.training.output_dir} already exists, using it")
    set_seed(args.common.seed)

    if args.common.debug:
        args.training.report_to = []

    # Determine main process: LOCAL_RANK is always set by accelerate/torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = (local_rank == 0)

    if is_main_process:
        if "trackio" in args.training.report_to:
            import trackio
            trackio.init(
                project=args.logging.trackio_project,
                space_id=args.logging.trackio_space_id,
                config=vars(args.training)
            )
            logger.info(f"Trackio initialized successfully")
        # wandb is managed by Trainer via report_to; no manual init needed

    return args

def train(
    config: TrainConfig = None
):
    # 0. parse args and prepare logger
    print(config)
    args = fuzzy_jobs(config)

    # 1. load tokenizer and dataset
    logger.info(f"Loading tokenizer from {args.model.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model.model_name_or_path)
    tokenizer.padding_side = "left"  # Configure for decoder-only architecture: use left padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "<|endoftext|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    
    logger.info(f"Loading dataset from {args.dataset.dataset_name_or_path}")
    dataset = load_dataset(
        args.dataset.dataset_name_or_path,
        example_numbers=args.dataset.example_numbers,
        tokenizer=tokenizer
    )
    train_dataset = dataset["train_dataset"]
    test_dataset = dataset["test_dataset"]
    reward_functions = dataset["reward_functions"]

    if "reward_weights" in dataset:
        reward_weights = dataset["reward_weights"]
    else:
        reward_weights = [1.0] * len(reward_functions)
    args.training.reward_weights = reward_weights

    # 2. load and configure model
    logger.info(f"Loading model from {args.model.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model.model_name_or_path,
        torch_dtype= torch.bfloat16 if args.model.dtype == "bfloat16" else torch.float16,
        attn_implementation="flash_attention_2"
    )
    logger.info(f"Model loaded successfully")

    # 3. configure peft
    optimizer = None
    if args.peft.use_peft:
        logger.info(f"Detected PEFT configuration, configuring {args.peft.type}")
        from perl.lora.adapter import apply_peft
        optimizer, model = apply_peft(model, args)
        logger.info(f"PEFT ({args.peft.type}) configured successfully")

    # 4.Training configuration
    training_args = GRPOConfig(
        **vars(args.training),
    )

    # 5.Train
    logger.info(f"Training model with GRPO")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=train_dataset,
        optimizers=(optimizer, None) if optimizer is not None else (None, None)
    )
    
    # 支持从 checkpoint 恢复训练
    resume_checkpoint = args.training.resume_from_checkpoint
    if resume_checkpoint == "true":
        resume_checkpoint = True
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    logger.info(f"Training completed successfully")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")
