<div align="center">

# PERL: Parameter-Efficient Reinforcement Learning  
> A minimal, modular, and lightning-fast framework for PEFT + RL.
</div>

## 🧩 Supported Parameter-Efficient Methods

| Method | Status | Description |
| :--- | :--- | :--- |
| **LoRA** | ✅ | Standard Low-Rank Adaptation |
| **DoRA** | ✅ | Weight-decomposed Low-Rank Adaptation |
| **MiSS** | ✅ | Mixture of Sub-Spaces (Efficient shard-sharing structure) |
| **AdaLoRA** | ✅ | Adaptive budget allocation for rank-adaptive matrices |
| **LoRA+** | ✅ | Differentiated learning rates for improved adaptation dynamics |
| **rsLORA** | ✅ | Rank stabilization scaling factors |
| **PiSSA** | ✅ | Principal Singular values & Singular vectors Adaptation |
| **MiLORA** | ✅ | Minor Singular components initialization |
| **LORA-FA** | ✅ | Memory-efficient adaptation with frozen projection matrix A |
| **VeRA** | ✅ | Vector-based Random Matrix Adaptation |
| **LN Tuning** | ✅ | Parameter-efficient tuning on Layer Normalization layers |
| **$IA^3$** | ✅ | Infused Adapter by Inhibiting and Amplifying Inner Activations |

## Environment Setup

```
uv pip install -r requirements.txt
```

```
uv pip install flash-attn --no-cache-dir --no-build-isolation
python -c "import flash_attn" # verify
```

## Training

```
source [your virtual env]/bin/activate
bash scripts/openr1/dapo_full.sh # run a full RL
bash scripts/openr1/dapo_lora.sh # run a lora RL
```