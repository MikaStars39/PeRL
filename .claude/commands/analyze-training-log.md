Analyze a SLIME async RL training log comprehensively: training health, entropy, reward, off-policy, collapse detection, and hyperparameter diagnosis. Save the analysis to a report file.

The log path is: $ARGUMENTS

Replace LOG_PATH in all commands below with the actual path provided above.

IMPORTANT: At the end, save the full analysis to a report file at `<LOG_DIR>/training_analysis_<TIMESTAMP>.md` where LOG_DIR is the directory containing the log file and TIMESTAMP is the current date/time (YYYYMMDD_HHMMSS). Print the report path when done.

Follow these steps IN ORDER. Run the bash commands to extract data, collect all output, then write both to stdout and to the report file.

## Step 1: Basic info and hyperparameters

```bash
echo "=== Log File Info ===" && wc -l LOG_PATH && ls -lh LOG_PATH && echo "" && echo "=== Key Hyperparameters ===" && head -3000 LOG_PATH | grep -E "(learning_rate|lr |adam_beta|weight_decay|entropy_coef|kl_coef|eps_clip|global_batch_size|rollout_batch_size|n_samples_per_prompt|rollout_max_response_len|advantage_estimator|custom_loss|use_tis|tis_clip|partial_rollout|mask_offpolicy|optimizer|colocate|use_distributed_optimizer)" | head -40
```

## Step 2: Extract per-step training metrics

```bash
grep "model.py:679 - step" LOG_PATH | python3 -c "
import sys, re

steps = []
for line in sys.stdin:
    data = line.strip()
    m_step = re.search(r'step (\d+):', data)
    if not m_step: continue
    step = int(m_step.group(1))
    def ext(k):
        m = re.search(r\"'\" + k + r\"':\s*([-\d.eE+nNaA]+)\", data)
        if not m: return None
        v = m.group(1)
        if 'nan' in v.lower(): return float('nan')
        return float(v)
    steps.append({
        'step': step,
        'pg_loss': ext('train/pg_loss'),
        'entropy': ext('train/entropy_loss'),
        'ppo_kl': ext('train/ppo_kl'),
        'ois': ext('train/ois'),
        'tis': ext('train/tis'),
        'logp_diff': ext('train/train_rollout_logprob_abs_diff'),
        'grad_norm': ext('train/grad_norm'),
        'pg_clipfrac': ext('train/pg_clipfrac'),
        'tis_clipfrac': ext('train/tis_clipfrac'),
    })

if not steps:
    print('ERROR: No training step metrics found')
    sys.exit(1)

# Print full table (sampled if >200 steps)
header = '{:>6} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}'.format(
    'step','pg_loss','entropy','ppo_kl','ois','tis','logp_diff','grad_norm')
print('=== Per-Step Training Metrics ===')
print(f'Total steps: {len(steps)}')
print(header); print('-'*85)

# Sample: first 10, then every N, then last 20
if len(steps) <= 60:
    sampled = steps
else:
    n = max(1, (len(steps)-30) // 30)
    sampled = steps[:10] + steps[10:-20:n] + steps[-20:]
    sampled = sorted(set([(s['step'], i) for i, s in enumerate(sampled)]))
    sampled = [steps[i] for _, i in sampled]

for s in sampled:
    print('{:>6} {:>10.4f} {:>10.4f} {:>10.6f} {:>8.4f} {:>8.4f} {:>10.6f} {:>10.4f}'.format(
        s['step'],
        s['pg_loss'] or 0, s['entropy'] or 0, s['ppo_kl'] or 0,
        s['ois'] or 0, s['tis'] or 0, s['logp_diff'] or 0, s['grad_norm'] or 0))
"
```

## Step 3: Extract per-rollout metrics

```bash
grep "rollout_data" LOG_PATH | python3 -c "
import sys, re

rollouts = []
for line in sys.stdin:
    data = line.strip()
    m_r = re.search(r'rollout_data (\d+)', data)
    if not m_r: continue
    r = int(m_r.group(1))
    def ext(k):
        m = re.search(r\"'\" + k + r\"':\s*([-\d.eE+nNaA]+)\", data)
        if not m: return None
        v = m.group(1)
        if 'nan' in v.lower(): return float('nan')
        return float(v)
    rollouts.append({
        'rollout': r,
        'raw_reward': ext('raw_reward'),
        'truncated': ext('truncated_ratio'),
        'resp_len': ext('response_length_mean'),
        'log_probs': ext('log_probs'),
        'rollout_log_probs': ext('rollout_log_probs'),
        'advantages': ext('advantages'),
    })

if not rollouts:
    print('ERROR: No rollout metrics found')
    sys.exit(1)

print('=== Per-Rollout Metrics ===')
print(f'Total rollouts: {len(rollouts)}')
header = '{:>6} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}'.format(
    'roll','reward','truncated','resp_len','log_probs','rollout_lp','lp_gap','advantages')
print(header); print('-'*100)

if len(rollouts) <= 40:
    sampled = rollouts
else:
    n = max(1, (len(rollouts)-15) // 25)
    sampled = rollouts[:5] + rollouts[5:-10:n] + rollouts[-10:]

for r in sampled:
    lp = r['log_probs'] or 0
    rlp = r['rollout_log_probs'] or 0
    gap = abs(lp - rlp)
    print('{:>6} {:>10.4f} {:>10.4f} {:>10.0f} {:>12.4f} {:>12.4f} {:>10.6f} {:>10.4f}'.format(
        r['rollout'], r['raw_reward'] or 0, r['truncated'] or 0, r['resp_len'] or 0,
        lp, rlp, gap, r['advantages'] or 0))
"
```

## Step 4: Extract rollout performance metrics (repetition, drops, etc.)

```bash
grep "rollout_perf" LOG_PATH | python3 -c "
import sys, re

perfs = []
for line in sys.stdin:
    data = line.strip()
    m_r = re.search(r'rollout_perf (\d+)', data)
    if not m_r: continue
    r = int(m_r.group(1))
    def ext(k):
        m = re.search(r\"'\" + k + r\"':\s*([-\d.eE+nNaA]+)\", data)
        if not m: return None
        v = m.group(1)
        if 'nan' in v.lower(): return float('nan')
        return float(v)
    perfs.append({
        'rollout': r,
        'drop0': ext('drop_zero_std_0'),
        'drop1': ext('drop_zero_std_1'),
        'rep_frac': ext('repetition_frac'),
        'trunc': ext('truncated_ratio'),
    })

if not perfs:
    print('No rollout_perf data found'); sys.exit(0)

print('=== Rollout Performance Metrics ===')
print(f'Total: {len(perfs)}')
header = '{:>6} {:>10} {:>10} {:>12} {:>10}'.format('roll','drop_0','drop_1','rep_frac','truncated')
print(header); print('-'*55)

if len(perfs) <= 40:
    sampled = perfs
else:
    n = max(1, (len(perfs)-10) // 25)
    sampled = perfs[:5] + perfs[5:-10:n] + perfs[-10:]

for p in sampled:
    print('{:>6} {:>10.0f} {:>10.0f} {:>12.4f} {:>10.4f}'.format(
        p['rollout'], p['drop0'] or 0, p['drop1'] or 0, p['rep_frac'] or 0, p['trunc'] or 0))
"
```

## Step 5: Comprehensive health diagnosis

```bash
grep "model.py:679 - step" LOG_PATH | python3 -c "
import sys, re, math

steps = []
for line in sys.stdin:
    data = line.strip()
    m = re.search(r'step (\d+):', data)
    if not m: continue
    step = int(m.group(1))
    def ext(k):
        m2 = re.search(r\"'\" + k + r\"':\s*([-\d.eE+nNaA]+)\", data)
        if not m2: return None
        v = m2.group(1)
        if 'nan' in v.lower(): return float('nan')
        return float(v)
    steps.append({'step': step, 'ois': ext('train/ois'), 'tis': ext('train/tis'),
        'entropy': ext('train/entropy_loss'), 'grad_norm': ext('train/grad_norm'),
        'pg_loss': ext('train/pg_loss'), 'ppo_kl': ext('train/ppo_kl'),
        'logp_diff': ext('train/train_rollout_logprob_abs_diff'),
        'is_ratio': ext('train/is_ratio_mean')})

if not steps: print('ERROR: No data'); sys.exit(1)
n = len(steps)

# Helper
def safe_vals(key):
    return [s[key] for s in steps if s[key] is not None and not math.isnan(s[key])]

ent = safe_vals('entropy')
grad = safe_vals('grad_norm')
ois = safe_vals('ois')
tis = safe_vals('tis')
is_ratio = safe_vals('is_ratio')
logp_diff = safe_vals('logp_diff')
kl = safe_vals('ppo_kl')

print('='*70)
print('           COMPREHENSIVE TRAINING HEALTH DIAGNOSIS')
print('='*70)
print(f'Total training steps analyzed: {n}')
print()

# --- 1. Entropy Analysis ---
print('--- 1. Entropy Analysis ---')
if ent:
    print(f'  Overall: mean={sum(ent)/len(ent):.4f}  min={min(ent):.6f}  max={max(ent):.4f}')
    # Check first vs last 20%
    first20 = ent[:max(1,len(ent)//5)]
    last20 = ent[-max(1,len(ent)//5):]
    f_avg = sum(first20)/len(first20)
    l_avg = sum(last20)/len(last20)
    decline = (f_avg - l_avg) / max(f_avg, 0.001) * 100
    print(f'  First 20%: mean={f_avg:.4f}  Last 20%: mean={l_avg:.4f}  Change: {-decline:+.1f}%')

    # Collapse detection
    near_zero = sum(1 for e in ent if e < 0.01)
    very_low = sum(1 for e in ent if e < 0.05)
    if near_zero > 0:
        first_zero = next(i for i, e in enumerate(ent) if e < 0.01)
        print(f'  CRITICAL: {near_zero} steps with entropy < 0.01 (first at step index {first_zero})')
        print(f'  DIAGNOSIS: ENTROPY COLLAPSE DETECTED')
    elif very_low > len(ent) * 0.1:
        print(f'  WARNING: {very_low} steps ({100*very_low/len(ent):.1f}%) with entropy < 0.05')
        print(f'  DIAGNOSIS: ENTROPY COLLAPSE RISK')
    elif decline > 60:
        print(f'  WARNING: Entropy declined {decline:.0f}% from start to end')
        print(f'  DIAGNOSIS: ENTROPY DECLINING - monitor closely')
    elif l_avg > 0.2:
        print(f'  DIAGNOSIS: ENTROPY STABLE')
    else:
        print(f'  DIAGNOSIS: ENTROPY LOW but not collapsed')
else:
    print('  No entropy data')

print()

# --- 2. IS Ratio / NaN Detection ---
print('--- 2. IS Ratio & NaN Detection ---')
nan_count = sum(1 for s in steps if s['is_ratio'] is not None and math.isnan(s['is_ratio']))
if nan_count > 0:
    first_nan = next(s['step'] for s in steps if s['is_ratio'] is not None and math.isnan(s['is_ratio']))
    print(f'  CRITICAL: {nan_count} steps with NaN IS ratio (first at step {first_nan})')
    print(f'  DIAGNOSIS: MODEL COLLAPSED (NaN in IS ratios)')
elif is_ratio:
    huge = sum(1 for v in is_ratio if v > 100)
    if huge > 0:
        print(f'  CRITICAL: {huge} steps with IS ratio > 100')
        print(f'  Max IS ratio: {max(is_ratio):.2e}')
        print(f'  DIAGNOSIS: IS RATIO EXPLOSION')
    else:
        print(f'  IS ratio: mean={sum(is_ratio)/len(is_ratio):.4f}  max={max(is_ratio):.4f}')
        print(f'  DIAGNOSIS: IS RATIOS HEALTHY')
else:
    print('  No IS ratio data (may use OIS instead)')

print()

# --- 3. Off-Policy Analysis ---
print('--- 3. Off-Policy Analysis ---')
if ois:
    ois_mean = sum(ois)/len(ois)
    severe = sum(1 for v in ois if v < 0.3)
    moderate = sum(1 for v in ois if v < 0.5)
    print(f'  OIS: mean={ois_mean:.4f}  min={min(ois):.4f}  max={max(ois):.4f}')
    print(f'  OIS<0.3 (severe off-policy): {severe}/{len(ois)} ({100*severe/len(ois):.1f}%)')
    print(f'  OIS<0.5 (moderate): {moderate}/{len(ois)} ({100*moderate/len(ois):.1f}%)')
    if logp_diff:
        print(f'  LogProb diff: mean={sum(logp_diff)/len(logp_diff):.6f}  max={max(logp_diff):.6f}')
    if tis:
        tis_mean = sum(tis)/len(tis)
        gap = abs(tis_mean - ois_mean)
        print(f'  TIS: mean={tis_mean:.4f}  TIS-OIS gap={gap:.6f}')
        if gap < 0.005:
            print(f'  TIS correction: INACTIVE (gap too small)')
        else:
            print(f'  TIS correction: ACTIVE')
    if ois_mean > 0.7:
        print(f'  DIAGNOSIS: OFF-POLICY HEALTHY')
    elif ois_mean > 0.4:
        print(f'  DIAGNOSIS: OFF-POLICY MODERATE (expected in async)')
    else:
        print(f'  DIAGNOSIS: OFF-POLICY SEVERE')
else:
    print('  No OIS data')

print()

# --- 4. Gradient Stability ---
print('--- 4. Gradient Stability ---')
if grad:
    print(f'  Grad norm: mean={sum(grad)/len(grad):.4f}  min={min(grad):.6f}  max={max(grad):.4f}')
    spikes = sum(1 for v in grad if v > 5.0)
    extreme = sum(1 for v in grad if v > 50.0)
    near_zero = sum(1 for v in grad if v < 0.01)
    if extreme > 0:
        print(f'  CRITICAL: {extreme} steps with grad_norm > 50 (max={max(grad):.1f})')
        print(f'  DIAGNOSIS: GRADIENT EXPLOSION')
    elif spikes > 0:
        print(f'  WARNING: {spikes} grad norm spikes > 5.0')
        print(f'  DIAGNOSIS: GRADIENT UNSTABLE')
    elif near_zero > len(grad) * 0.1:
        print(f'  WARNING: {near_zero} steps with grad_norm < 0.01')
        print(f'  DIAGNOSIS: GRADIENT VANISHING (model may be frozen)')
    else:
        # Check trend
        first_g = grad[:max(1,len(grad)//5)]
        last_g = grad[-max(1,len(grad)//5):]
        print(f'  First 20%: mean={sum(first_g)/len(first_g):.4f}  Last 20%: mean={sum(last_g)/len(last_g):.4f}')
        print(f'  DIAGNOSIS: GRADIENT STABLE')
else:
    print('  No grad norm data')

print()

# --- 5. Reward Learning Signal ---
print('--- 5. Learning Signal ---')
pg = safe_vals('pg_loss')
if pg:
    first_pg = pg[:max(1,len(pg)//5)]
    last_pg = pg[-max(1,len(pg)//5):]
    print(f'  PG loss: first 20% mean={sum(first_pg)/len(first_pg):.4f}  last 20% mean={sum(last_pg)/len(last_pg):.4f}')
    if all(v > -0.01 for v in last_pg):
        print(f'  WARNING: PG loss near zero or positive in last 20% -- weak/no learning signal')
    elif sum(last_pg)/len(last_pg) < sum(first_pg)/len(first_pg) - 0.05:
        print(f'  PG loss becoming more negative -- model is learning')
    else:
        print(f'  PG loss stable -- model may be stagnating')

print()

# --- 6. Overall Verdict ---
print('='*70)
print('OVERALL VERDICT:')
has_nan = nan_count > 0
has_is_explosion = is_ratio and max(is_ratio) > 100
has_entropy_collapse = ent and sum(1 for e in ent if e < 0.01) > 0
has_grad_explosion = grad and max(grad) > 50
has_grad_vanish = grad and sum(1 for v in grad if v < 0.01) > len(grad) * 0.1
has_entropy_decline = ent and len(ent) > 20 and sum(ent[-max(1,len(ent)//5):])/max(1,len(ent)//5) < 0.05

if has_nan or has_is_explosion:
    print('  STATUS: COLLAPSED (IS ratio NaN/explosion)')
    print('  Likely cause: optimizer too aggressive for async off-policy')
    print('  Fix: reduce lr, increase epsilon (for Roo/Muon), tighten IS clipping')
elif has_entropy_collapse or has_entropy_decline:
    print('  STATUS: COLLAPSED (entropy collapse)')
    print('  Likely cause: no entropy regularization (entropy_coef=0.0)')
    print('  Fix: add --entropy-coef 0.001~0.01')
elif has_grad_explosion:
    print('  STATUS: UNSTABLE (gradient explosion)')
    print('  Likely cause: policy diverged, possibly from entropy collapse or IS ratio issues')
    print('  Fix: reduce lr, add entropy regularization, tighten clipping')
elif has_grad_vanish:
    print('  STATUS: FROZEN (gradient vanishing)')
    print('  Likely cause: entropy collapsed to near-zero, policy is deterministic')
    print('  Fix: add --entropy-coef, increase lr')
elif ent and sum(ent)/len(ent) > 0.2 and ois and sum(ois)/len(ois) > 0.3:
    # Check if reward is improving
    print('  STATUS: HEALTHY (no collapse, stable metrics)')
    if pg and sum(pg[-max(1,len(pg)//5):])/max(1,len(pg)//5) > 0:
        print('  NOTE: but reward may be stagnating (pg_loss >= 0)')
else:
    print('  STATUS: AT RISK (low entropy or high off-policy)')
print('='*70)
"
```

## Step 6: Timing breakdown

```bash
grep "Timer" LOG_PATH | grep "elapsed" | grep -o "Timer [a-z_]* end (elapsed: [0-9.]*s)" | python3 -c "
import sys, re
from collections import defaultdict
timers = defaultdict(list)
for line in sys.stdin:
    m = re.match(r'Timer (\w+) end \(elapsed: ([\d.]+)s\)', line.strip())
    if m: timers[m.group(1)].append(float(m.group(2)))
print('=== Timing Breakdown ===')
for name in ['train_wait','actor_train','log_probs','update_weights','data_preprocess','train']:
    v = timers.get(name,[])
    if v: print(f'{name:>20}: mean={sum(v)/len(v):>8.1f}s  min={min(v):>8.1f}s  max={max(v):>8.1f}s  n={len(v)}')
tw=timers.get('train_wait',[]); at=timers.get('actor_train',[])
if tw and at:
    total=sum(tw)+sum(at)
    print(f'  GPU utilization: {100*sum(at)/total:.1f}% training, {100*sum(tw)/total:.1f}% idle/waiting')
"
```

## Step 7: Error and anomaly scan

```bash
echo "=== Error & Anomaly Scan ===" && echo "Aborted groups: $(grep -c 'Returned aborted group' LOG_PATH 2>/dev/null || echo 0)" && echo "No-progress warnings: $(grep -c 'No progress for' LOG_PATH 2>/dev/null || echo 0)" && echo "NaN occurrences in metrics: $(grep 'model.py:679' LOG_PATH | grep -c 'nan' 2>/dev/null || echo 0)" && echo "CUDA OOM: $(grep -c 'CUDA out of memory' LOG_PATH 2>/dev/null || echo 0)" && echo "Tracebacks: $(grep -c 'Traceback' LOG_PATH 2>/dev/null || echo 0)" && echo "RuntimeErrors: $(grep -c 'RuntimeError' LOG_PATH 2>/dev/null || echo 0)" && echo "Failed requests: $(grep -c 'Failed to send' LOG_PATH 2>/dev/null || echo 0)" && echo "KV cache full: $(grep -c 'KV cache pool is full' LOG_PATH 2>/dev/null || echo 0)"
```

## Step 8: Eval results (if available)

```bash
LOG_DIR=$(dirname LOG_PATH)
if ls "$LOG_DIR"/eval_results_* 2>/dev/null | head -1 > /dev/null 2>&1; then
    echo "=== Eval Results ==="
    for f in "$LOG_DIR"/eval_results_*; do
        echo "--- $(basename $f) ---"
        cat "$f" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, dict):
        for k, v in sorted(data.items()):
            if isinstance(v, (int, float)):
                print(f'  {k}: {v:.4f}')
            elif isinstance(v, dict):
                for k2, v2 in sorted(v.items()):
                    if isinstance(v2, (int, float)):
                        print(f'  {k}/{k2}: {v2:.4f}')
except: print('  (could not parse)')
" 2>/dev/null
    done
else
    echo "=== Eval Results: None found ==="
fi
```

## Step 9: Write report file

After collecting all output from Steps 1-8, write a complete markdown report file to `<LOG_DIR>/training_analysis_<TIMESTAMP>.md`. The report must contain:

1. Header with log path, model name, and analysis timestamp
2. Hyperparameter summary table from Step 1
3. Per-step training metrics table (sampled) from Step 2
4. Per-rollout metrics table (sampled) from Step 3
5. Rollout performance (repetition, drops) from Step 4
6. The full health diagnosis from Step 5
7. Timing breakdown from Step 6
8. Error scan from Step 7
9. Eval results from Step 8 (if any)
10. A Chinese diagnosis section (诊断报告) covering:
    - **训练状态**: 健康/停滞/崩溃/风险中
    - **Entropy 分析**: 是否崩溃？趋势如何？是否需要 entropy regularization？
    - **Off-Policy 分析**: OIS 分布、log_prob gap、TIS 是否生效
    - **IS Ratio 分析**: 是否有爆炸或 NaN
    - **梯度稳定性**: grad norm 趋势、是否有突刺或消失
    - **Reward 趋势**: 是否在学习？是否停滞或退化？
    - **Rollout 质量**: 重复率、截断率、drop_zero_std 趋势
    - **GPU 利用率**: train_wait 占比
    - **具体建议**: 需要调整的超参数及建议值

Print the report file path at the end so the user can find it.
