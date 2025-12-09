# copyright (c) 2025, mikastars39.org
# All rights reserved.
# This source code is licensed under the Apache-2.0 License.
# See the LICENSE file in the root directory for details.

from .slicefine import register_slicefine_method
register_slicefine_method() # register slicefine method to peft

def apply_lora(model, args):
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

def apply_dora(model, args):
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

def apply_vera(model, args):
    from peft import VeraConfig, get_peft_model
    config = VeraConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_miss(model, args):
    from peft import MissConfig, get_peft_model
    config = MissConfig(r=args.peft.r)
    return None, get_peft_model(model, config)

def apply_pissa(model, args):
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

def apply_milora(model, args):
    from .milora import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_layernorm(model, args):
    from peft import get_peft_model, TaskType, LNTuningConfig
    peft_config = LNTuningConfig(
        task_type=TaskType.CAUSAL_LM,
    )
    return None, get_peft_model(model, peft_config)

def apply_adalora(model, args):
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

def apply_IA3(model, args):
    from peft import IA3Config, get_peft_model, TaskType
    config = IA3Config(task_type=TaskType.CAUSAL_LM)
    return None, get_peft_model(model, config)

def apply_milora_plus(model, args):
    from .milora_plus import add_svd_initialized_lora
    return None, add_svd_initialized_lora(
        model=model,
        rank=args.peft.r,
    )

def apply_lorafa(model, args):
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

def apply_lora_plus(model, args):
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

def apply_slicefine(model, args):
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

def apply_hra(model, args):
    from peft import HRAConfig, get_peft_model
    config = HRAConfig(
        r=args.peft.r,
        target_modules=args.peft.target_modules,
        init_weights="True",
    )
    return None, get_peft_model(model, config)

# ------------------------------ mapping function to peft type ------------------------------

PEFT_TYPE_TO_FUNCTION_MAPPING = {
    "lora": apply_lora,
    "dora": apply_dora,
    "slicefine": apply_slicefine,
    "milora": apply_milora,
    "milora_plus": apply_milora_plus,
    "lorafa": apply_lorafa,
    "lora_plus": apply_lora_plus,
    "adalora": apply_adalora,
    "IA3": apply_IA3,
    "layernorm": apply_layernorm,
    "vera": apply_vera,
    "miss": apply_miss,
    "pissa": apply_pissa,
    "hra": apply_hra,
}

# ------------------------------ dispatch function to peft type ------------------------------

def apply_peft(model, args):
    """Dispatch function that routes to the appropriate PEFT method based on args.peft.type"""
    peft_type = args.peft.type
    if peft_type in PEFT_TYPE_TO_FUNCTION_MAPPING:
        return PEFT_TYPE_TO_FUNCTION_MAPPING[peft_type](model, args)
    else:
        raise ValueError(f"Unsupported PEFT type: {peft_type}")