# PeRL - Parameter-Efficient Reinforcement Learning

A minimal, modular framework for PEFT (Parameter-Efficient Fine-Tuning) + RL (Reinforcement Learning), and Optimizer design.

## Repository Structure

```
PeRL/
├── modules/          # Core modules (contains submodules + local code)
│   ├── trl/          # Local TRL-based training code (modules/trl/perl/)
│   ├── slime/        # [submodule] SLIME training backend
│   ├── Megatron-LM/  # [submodule] Megatron-LM (forked for PeRL)
│   └── Emerging-Optimizers/  # [submodule] NVIDIA NeMo optimizers
├── recipes/          # Training scripts & configs
│   ├── trl/          # TRL recipes (openr1/, openmath/, accelerate/, a6000/)
│   └── slime/        # SLIME recipes
├── env/              # Environment setup (requirements_hard.txt)
├── doc/              # Documentation
└── assets/           # Images and logos
```

## Key Concepts

- **modules/trl/perl/**: Core PeRL implementation on top of TRL. Contains PEFT method integrations (LoRA, DoRA, AdaLoRA, MiSS, PiSSA, VeRA, etc.).
- **recipes/**: Shell scripts for launching training. Organized by backend (trl/slime) and dataset/config (openr1, openmath).
- **Submodules**: `slime`, `Megatron-LM`, `Emerging-Optimizers` are git submodules under `modules/`.

## TODO

IMPORTANT: read the doc/TODO.md for what we will do for this project
