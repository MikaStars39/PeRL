Analyze the training log for train-inference inconsistency (推训不一致) issues in SLIME async RL training. Save the analysis result to a report file.

The log path is: $ARGUMENTS

Replace LOG_PATH in all commands below with the actual path provided above.

IMPORTANT: At the end, save the full analysis to a report file at `<LOG_DIR>/offpolicy_report_<TIMESTAMP>.md` where LOG_DIR is the directory containing the log file and TIMESTAMP is the current date/time (YYYYMMDD_HHMMSS). Print the report path when done.

Follow these steps IN ORDER. Run the bash commands to extract data, collect all output, then write both to stdout and to the report file.

## Step 1: Extract per-step training metrics table

Run this command (replace LOG_PATH with the actual log path):

```bash
grep "model.py:679 - step" LOG_PATH | sed "s/.*step \([0-9]*\): {\(.*\)}/\1 \2/" | python3 -c "
import sys, re
header = '{:>5} {:>10} {:>10} {:>10} {:>8} {:>8} {:>10} {:>10}'.format('step','pg_loss','entropy','clipfrac','ois','tis','logp_diff','grad_norm')
print(header); print('-'*85)
for line in sys.stdin:
    parts = line.strip().split(' ', 1)
    if len(parts)<2: continue
    step = int(parts[0]); data = parts[1]
    def ext(k):
        m = re.search(r\"'\" + k + r\"':\s*([-\d.eE+]+)\", data)
        return float(m.group(1)) if m else 0
    print('{:>5} {:>10.4f} {:>10.4f} {:>10.6f} {:>8.4f} {:>8.4f} {:>10.6f} {:>10.4f}'.format(
        step, ext('train/pg_loss'), ext('train/entropy_loss'), ext('train/pg_clipfrac'),
        ext('train/ois'), ext('train/tis'), ext('train/train_rollout_logprob_abs_diff'), ext('train/grad_norm')))
"
```

## Step 2: Compute off-policy summary statistics + diagnosis

```bash
grep "model.py:679 - step" LOG_PATH | python3 -c "
import sys, re
ois_vals, tis_vals, entropy_vals, grad_norms = [], [], [], []
for line in sys.stdin:
    data = line.strip()
    def ext(k):
        m = re.search(r\"'\" + k + r\"':\s*([-\d.eE+]+)\", data)
        return float(m.group(1)) if m else None
    o=ext('train/ois'); t=ext('train/tis'); e=ext('train/entropy_loss'); g=ext('train/grad_norm')
    if o is not None: ois_vals.append(o)
    if t is not None: tis_vals.append(t)
    if e is not None: entropy_vals.append(e)
    if g is not None: grad_norms.append(g)
if not ois_vals: print('ERROR: No step metrics found'); sys.exit(1)

print('=== Off-Policy Summary ===')
print(f'Total steps: {len(ois_vals)}')
print()
print('--- OIS (On-policy IS ratio, ideal=1.0) ---')
print(f'  Mean: {sum(ois_vals)/len(ois_vals):.4f}  Min: {min(ois_vals):.4f}  Max: {max(ois_vals):.4f}')
sev=sum(1 for v in ois_vals if v<0.5); mod=sum(1 for v in ois_vals if v<0.8)
print(f'  OIS<0.5 (severe): {sev}/{len(ois_vals)} ({100*sev/len(ois_vals):.1f}%)')
print(f'  OIS<0.8 (moderate): {mod}/{len(ois_vals)} ({100*mod/len(ois_vals):.1f}%)')
print()
print('--- TIS (Token IS) ---')
tis_mean=sum(tis_vals)/len(tis_vals); ois_mean=sum(ois_vals)/len(ois_vals)
print(f'  Mean: {tis_mean:.4f}  TIS-OIS gap: {abs(tis_mean-ois_mean):.6f}')
print()
print('--- Entropy ---')
print(f'  Mean: {sum(entropy_vals)/len(entropy_vals):.4f}  Min: {min(entropy_vals):.4f}  Max: {max(entropy_vals):.4f}')
low=sum(1 for v in entropy_vals if v<0.3)
print(f'  Entropy<0.3 (collapse risk): {low}/{len(entropy_vals)} ({100*low/len(entropy_vals):.1f}%)')
print()
print('--- Grad Norm ---')
print(f'  Mean: {sum(grad_norms)/len(grad_norms):.4f}  Max: {max(grad_norms):.4f}')
spk=sum(1 for v in grad_norms if v>1.0)
print(f'  grad_norm>1.0 (spikes): {spk}/{len(grad_norms)} ({100*spk/len(grad_norms):.1f}%)')
if spk>0:
    spike_steps=[i for i,v in enumerate(grad_norms) if v>1.0]
    print(f'  Spike at step indices: {spike_steps[:10]}')
print()

# Periodicity detection
print('--- Periodicity (sawtooth) ---')
drops=[]
for i in range(1, len(ois_vals)):
    if ois_vals[i]<ois_vals[i-1]*0.7 and ois_vals[i]<0.6: drops.append(i)
if len(drops)>=2:
    intervals=[drops[i]-drops[i-1] for i in range(1,len(drops))]
    print(f'  {len(drops)} major OIS drops, avg interval: {sum(intervals)/len(intervals):.1f} steps')
else:
    print(f'  No clear periodic pattern ({len(drops)} drops)')

# Severity classification
print()
print('=== DIAGNOSIS ===')
if ois_mean>0.8 and 100*sev/len(ois_vals)<10:
    print('  Off-policy severity: HEALTHY')
elif ois_mean>=0.5:
    print('  Off-policy severity: MODERATE')
else:
    print('  Off-policy severity: SEVERE')
if abs(tis_mean-ois_mean)<0.01:
    print('  TIS correction: INEFFECTIVE (TIS ~= OIS)')
else:
    print('  TIS correction: ACTIVE')
if low/len(entropy_vals)>0.3:
    print('  Entropy: WARNING - frequent low entropy, possible collapse')
elif max(entropy_vals)/max(0.01,min(entropy_vals))>3:
    print('  Entropy: UNSTABLE - large sawtooth oscillation')
else:
    print('  Entropy: STABLE')
if spk>0:
    print(f'  Gradient: UNSTABLE - {spk} spikes detected')
else:
    print('  Gradient: STABLE')
"
```

## Step 3: Timing breakdown

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
    print(f'  GPU idle ratio: {100*sum(tw)/total:.1f}% waiting, {100*sum(at)/total:.1f}% training')
"
```

## Step 4: Aborted rollout groups

```bash
echo "=== Aborted Rollouts ===" && echo "Total aborted groups: $(grep -c 'Returned aborted group' LOG_PATH)" && echo "Weight update cycles: $(grep -c 'slime-pp_0.*Update weights: 0it' LOG_PATH)"
```

## Step 5: Write report file

After collecting all output from Steps 1-4, write a complete markdown report file to `<LOG_DIR>/offpolicy_report_<TIMESTAMP>.md`. The report must contain:

1. Header with log path and analysis timestamp
2. The full per-step metrics table from Step 1
3. The summary statistics from Step 2
4. The timing breakdown from Step 3
5. The aborted rollout counts from Step 4
6. A Chinese diagnosis section covering:
   - **Off-Policy 严重程度**: 基于 OIS 分布 (健康/中等/严重)
   - **Entropy 稳定性**: 锯齿形波动？collapse 风险？
   - **TIS 有效性**: TIS 是否在纠正 off-policy bias？
   - **梯度稳定性**: grad norm 突刺？与 off-policy 的关联？
   - **GPU 利用率**: train_wait 占比，GPU 空闲浪费
   - **Rollout 浪费**: 每个 cycle 的 abort 数量，推理算力浪费
   - **具体建议**: 可调参数及建议值 (update-weights-interval, mask-offpolicy, rollout-max-response-len 等)

Print the report file path at the end so the user can find it.
