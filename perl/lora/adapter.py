def apply_lora(model, args):

    from .slicefine import register_slicefine_method
    register_slicefine_method() # register slicefine method to peft

    if args.peft.type == "lora":
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            peft_type="LORA",
            task_type=args.peft.task_type,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            target_modules=args.peft.target_modules,
            lora_dropout=args.peft.lora_dropout,
        )
        return None, get_peft_model(model, config)
    elif args.peft.type == "dora":
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(
            peft_type="LORA",
            use_dora=True,
            task_type=args.peft.task_type,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            target_modules=args.peft.target_modules,
            lora_dropout=args.peft.lora_dropout,
        )
        return None, get_peft_model(model, config)
    elif args.peft.type == "vera":
        from peft import VeraConfig, get_peft_model
        config = VeraConfig(r=args.peft.r)
        return None, get_peft_model(model, config)
    elif args.peft.type == "miss":
        from peft import MissConfig, get_peft_model
        config = MissConfig(r=args.peft.r)
        return None, get_peft_model(model, config)
    elif args.peft.type == "pissa":
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            # init_lora_weights="pissa", # Configure the initialization method to "pissa", which may take several minutes to execute SVD on the pre-trained model.
            init_lora_weights="pissa_niter_4", # Initialize the PiSSA with fast SVD, which completes in just a few seconds.
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            lora_dropout=args.peft.lora_dropout, # Since the component of the PiSSA adapter are the principal singular values and vectors, dropout should be set to 0 to avoid random discarding.
            target_modules=args.peft.target_modules,
            task_type=args.peft.task_type,
        )
        return None, get_peft_model(model, lora_config)
    elif args.peft.type == "milora":
        from .milora import add_svd_initialized_lora
        return None, add_svd_initialized_lora(
            model=model,
            rank=args.peft.r,
        )
    elif args.peft.type == "layernorm":
        from peft import get_peft_model, TaskType, LNTuningConfig
        peft_config = LNTuningConfig(
            task_type=TaskType.CAUSAL_LM,
        )
        return None, get_peft_model(model, peft_config)
    elif args.peft.type == "adalora":
        from peft import AdaLoraConfig, get_peft_model
        config = AdaLoraConfig(
            peft_type="ADALORA",
            task_type=args.peft.task_type,
            init_r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            target_modules=args.peft.target_modules,
            lora_dropout=args.peft.lora_dropout,
            total_step=args.training.max_steps,
        )
        return None, get_peft_model(model, config)
    elif args.peft.type == "IA3":
        from peft import IA3Config, get_peft_model, TaskType
        config = IA3Config(task_type=TaskType.CAUSAL_LM)
        return None, get_peft_model(model, config)
    elif args.peft.type == "milora_plus":
        from .milora_plus import add_svd_initialized_lora
        return None, add_svd_initialized_lora(
            model=model,
            rank=args.peft.r,
        )
    elif args.peft.type == "lorafa":
        from peft import LoraConfig, get_peft_model
        from peft.optimizers import create_lorafa_optimizer

        config = LoraConfig(
            peft_type="LORA",
            task_type=args.peft.task_type,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            target_modules=args.peft.target_modules,
            lora_dropout=args.peft.lora_dropout,
        ) 

        model = get_peft_model(model, config)
            
        optimizer = create_lorafa_optimizer(
            model=model,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            lr=args.training.learning_rate,
        )
        return optimizer, model
    elif args.peft.type == "lora_plus":
        from peft import LoraConfig, get_peft_model
        from torch.optim import AdamW
        from peft.optimizers import create_loraplus_optimizer
        
        # First create the LoRA model
        config = LoraConfig(
            peft_type="LORA",
            task_type=args.peft.task_type,
            r=args.peft.r,
            lora_alpha=args.peft.lora_alpha,
            target_modules=args.peft.target_modules,
            lora_dropout=args.peft.lora_dropout,
        )
        model = get_peft_model(model, config)
        
        # Then create the LoraPlus optimizer with different learning rates for A and B
        optimizer = create_loraplus_optimizer(
            model=model,
            optimizer_cls=AdamW,
            lr=args.training.learning_rate,
            loraplus_lr_ratio=2.0,
        )
        return optimizer, model
    
    elif args.peft.type == "slicefine":
        from .slicefine import SliceFineConfig
        from peft import get_peft_model
        config = SliceFineConfig(
            r=args.peft.r,
            slice_mode=getattr(args.peft, "slice_mode", "column"),
            slice_position=getattr(args.peft, "slice_position", 0),
            target_modules=args.peft.target_modules,
            bias="all" if getattr(args.peft, "bias", False) else "none"
        )
        print(f"[SliceFine] Applying SliceFine with rank={config.r}, modules={config.target_modules}")
        
        peft_model = get_peft_model(model, config)
        
        peft_model.print_trainable_parameters()
        
        trainable_params = [p for p in peft_model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError(
                "[SliceFine Error] No trainable parameters found! \n"
                "1. Check if 'target_modules' match the model architecture.\n"
                "2. Check if 'part_T' is correctly set to requires_grad=True."
            )
        return None, peft_model

    else:
        raise ValueError(f"Unsupported PEFT type: {args.peft.type}")