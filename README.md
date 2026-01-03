<div align="center">
<img src="https://pbs.twimg.com/media/G9uYc9HasAAWCm5?format=jpg&name=medium" alt="logo" width="600" margin="10px"></img>
  
# PERL: Parameter-Efficient Reinforcement Learning  
> A minimal, modular, and lightning-fast framework for PEFT + RL.

| [**AlphaXiv**](https://www.alphaxiv.org/abs/2512.23165)
| [**ArXiv**](https://www.arxiv.org/abs/2512.23165)
| [**Checkpoints**](https://huggingface.co/MikaStars39/PeRL)
| [**Wandb Log**](https://wandb.ai/mikastars-zhejiang-university/PeRL_logs)

</div>

## Supported Parameter-Efficient Methods

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
