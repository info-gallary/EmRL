"""Auto-restarting wrapper for Phase 9 (survives transient CUDA errors).
BC warmup is checkpointed (bc_warmup_k16_v2.pt) so restarts skip it and resume PPO."""
import io, sys, time, subprocess
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

MAX_RESTARTS = 25
LOG = Path('results/gpu_phase9.log'); LOG.parent.mkdir(exist_ok=True)

def best():
    c = list(Path('checkpoints/phase9').glob('best_model_bdr*.pt'))
    b = 0.0
    for p in c:
        try: b = max(b, float(p.stem.replace('best_model_bdr', '')))
        except ValueError: pass
    return b

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"\n[RESILIENT-P9] attempt {attempt}/{MAX_RESTARTS} best={best():.4f}", flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        p = subprocess.run([sys.executable, 'scripts/training/run_phase9_train.py'],
                           stdout=f, stderr=subprocess.STDOUT,
                           encoding='utf-8', errors='replace')
    if p.returncode == 0:
        print(f"[RESILIENT-P9] complete. best={best():.4f}", flush=True)
        break
    print(f"[RESILIENT-P9] crash exit={p.returncode} best={best():.4f}, cooling 30s", flush=True)
    time.sleep(30)
