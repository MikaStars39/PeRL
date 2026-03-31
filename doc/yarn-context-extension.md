# YaRN Context Extension in Slime SFT

## Overview

YaRN (Yet another RoPE extensioN) allows extending a model's context length beyond its pretrained `max_position_embeddings` during fine-tuning, without retraining from scratch. This document records the implementation details and architecture-level understanding gained while adding YaRN support for MLA (Multi-Latent Attention) models (e.g., Moonlight-16B-A3B) in the slime training framework.

## Architecture: How YaRN Flows Through the Stack

```
CLI args
  │
  ├─ slime/utils/arguments.py          # registers --original-max-position-embeddings, --beta-fast, --beta-slow
  ├─ slime/backends/megatron_utils/arguments.py  # _set_default_megatron_args: auto-infers scaling_factor, sets rope_type="yarn"
  │
  ▼
Megatron-LM arguments.py
  │  core_transformer_config_from_args():
  │    for f in dataclasses.fields(config_class):
  │        if hasattr(args, f.name):
  │            kw_args[f.name] = getattr(args, f.name)
  │
  │  When multi_latent_attention=True, config_class = MLATransformerConfig
  │  MLATransformerConfig has fields:
  │    rope_type, rotary_scaling_factor, original_max_position_embeddings,
  │    beta_fast, beta_slow, mscale, mscale_all_dim, rotary_base
  │
  ▼
MLA attention (multi_latent_attention.py)
  │  if config.rope_type == "yarn":
  │      self.rotary_pos_emb = YarnRotaryEmbedding(
  │          scaling_factor=config.rotary_scaling_factor,
  │          original_max_position_embeddings=config.original_max_position_embeddings,
  │          beta_fast=config.beta_fast, beta_slow=config.beta_slow, ...)
  │
  ▼
YarnRotaryEmbedding (megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py)
  │  Applies frequency-domain interpolation with correction range
  │  Computes mscale concentration factor for attention scaling
```

### Key Distinction: MLA vs Non-MLA Path

- **MLA models** (DeepSeek V3, Moonlight): RoPE is handled *inside* `MultiLatentAttention.__init__`, which reads `config.rope_type` and creates `YarnRotaryEmbedding` directly. The `GPTModel` does NOT create rotary embeddings for MLA.
- **Non-MLA models** (Qwen, Llama): RoPE is handled in `GPTModel.__init__`, which checks `position_embedding_type == 'yarn'` and reads `config.yarn_*` fields (prefixed). This path requires different field names (`yarn_rotary_scaling_factor` etc.) and is only fully wired for `MLATransformerConfig`.

For non-MLA models wanting YaRN, additional work would be needed:
1. Add `yarn_*` fields to base `TransformerConfig`
2. Add `'yarn'` to `--position-embedding-type` choices in Megatron arguments
3. Remove the assertion that non-MLA only supports `rope_type="rope"`

## What We Changed

### slime (on dev branch)

1. **`slime/utils/arguments.py`** - Added `add_yarn_arguments()`:
   - `--original-max-position-embeddings`: the pretrained model's original context length
   - `--beta-fast` (default 32.0): controls which RoPE dimensions are interpolated vs extrapolated
   - `--beta-slow` (default 1.0): lower bound for the correction range

2. **`slime/backends/megatron_utils/arguments.py`** - Extended `_set_default_megatron_args()`:
   - When `original_max_position_embeddings` is set and `seq_length` exceeds it, automatically infers `rotary_scaling_factor` and sets `rope_type = "yarn"`

### PeRL (on main branch)

3. **`recipes/slime/sft/run-moonlight-16B-A3B-sft-muon-yarn-32k.sh`** - New recipe:
   - `YARN_ARGS` block with `--original-max-position-embeddings 8192 --rotary-scaling-factor 4.0`
   - `--rollout-max-context-len 32000` (up from 8000)
   - `--seq-length` is implicitly set via `--rollout-max-context-len` in slime's data flow

## Existing Megatron CLI Args (no changes needed)

These args already exist in Megatron and are auto-mapped to `MLATransformerConfig`:

| CLI arg | MLATransformerConfig field | Default | Description |
|---|---|---|---|
| `--rotary-scaling-factor` | `rotary_scaling_factor` | 1.0 | YaRN scaling factor = target_len / original_len |
| `--mscale` | `mscale` | 1.0 | Attention concentration factor |
| `--mscale-all-dim` | `mscale_all_dim` | 0.0 | Mscale across all dims |
| `--rotary-base` | `rotary_base` | 10000 | RoPE base frequency |
| `--rope-type` | `rope_type` | "yarn" (MLA default) | "rope" or "yarn" |

## Args We Added to slime

| CLI arg | MLATransformerConfig field | Default | Description |
|---|---|---|---|
| `--original-max-position-embeddings` | `original_max_position_embeddings` | None (4096 in config) | Model's pretrained max context |
| `--beta-fast` | `beta_fast` | 32.0 | Upper bound for YaRN correction range |
| `--beta-slow` | `beta_slow` | 1.0 | Lower bound for YaRN correction range |

## YaRN Parameter Guidelines

### scaling_factor

`scaling_factor = target_context_length / original_max_position_embeddings`

| Model | Original | Target | Factor |
|---|---|---|---|
| Moonlight-16B-A3B | 8192 | 32768 | 4.0 |
| DeepSeek-V3 | 4096 | 163840 | 40.0 |

### beta_fast / beta_slow

These control which RoPE frequency dimensions are interpolated (low freq) vs kept (high freq):
- `beta_fast = 32` and `beta_slow = 1` are the standard defaults from the YaRN paper
- Higher `beta_fast` preserves more high-frequency dimensions (short-range attention)
- These rarely need tuning

### mscale / mscale_all_dim

These control the attention logit scaling compensation:
- `mscale = 1.0, mscale_all_dim = 0.0` is standard (no extra scaling)
- `mscale = 1.0, mscale_all_dim = 1.0` is what Moonlight uses (applies scaling across all dims)
- DeepSeek-V3 uses `mscale = 0.707, mscale_all_dim = 0.707`

## Moonlight-16B-A3B Model Reference

From `config.json`:
- Architecture: `DeepseekV3ForCausalLM` (MLA)
- `max_position_embeddings`: 8192
- `rope_theta`: 50000
- `qk_rope_head_dim`: 64 (position embedding dimension in MLA)
- `qk_nope_head_dim`: 128
- `kv_lora_rank`: 512
- `q_lora_rank`: null (no query compression, unlike DeepSeek-V3)
- `hidden_size`: 2048, `num_hidden_layers`: 27
- `n_routed_experts`: 64, `num_experts_per_tok`: 6

## Potential Issues and Notes

1. **Memory**: Longer sequences require significantly more memory. The recipe keeps `--max-tokens-per-gpu 32000` and `--recompute-granularity full` to manage this. If OOM occurs, increase `--recompute-num-layers` or reduce `--max-tokens-per-gpu`.

2. **seq_length vs max_position_embeddings**: In slime's `_set_default_megatron_args`, `args.max_position_embeddings = args.seq_length`. When `seq_length` is not explicitly set, it defaults to 4096. The `--rollout-max-context-len` in the SFT recipe controls the actual data length but does not set `--seq-length` directly. If training data can exceed 32k, explicitly pass `--seq-length 32768`.

3. **HF config validation**: `_hf_validate_args` checks that Megatron args match the HF config. Since we're intentionally changing `max_position_embeddings` beyond the HF config value, this validation won't fail because it doesn't check `max_position_embeddings` — it only checks hidden_size, num_layers, etc.

4. **Inference (SGLang)**: YaRN parameters only affect the Megatron training side. If using SGLang for inference/rollout (non debug-train-only mode), you would also need to configure SGLang's RoPE scaling separately via its own config.
