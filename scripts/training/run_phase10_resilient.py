"""Auto-restarting wrapper for Phase 10 (multi-scale). Survives transient CUDA errors.
BC warmup is checkpointed (bc_warmup_multiscale.pt) so restarts skip it and resume PPO."""
import io, sys, time, subprocess
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

MAX_RESTARTS = 25
LOG = Path('results/gpu_phase10.log'); LOG.parent.mkdir(exist_ok=True)

def best():
    c = list(Path('checkpoints/phase10').glob('best_model_bdr*.pt'))
    b = 0.0
    for p in c:
        try: b = max(b, float(p.stem.replace('best_model_bdr', '')))
        except ValueError: pass
    return b

for attempt in range(1, MAX_RESTARTS + 1):
    print(f"\n[RESILIENT-P10] attempt {attempt}/{MAX_RESTARTS} best={best():.4f}", flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        p = subprocess.run([sys.executable, 'scripts/training/run_phase10_train.py'],
                           stdout=f, stderr=subprocess.STDOUT,
                           encoding='utf-8', errors='replace')
    if p.returncode == 0:
        print(f"[RESILIENT-P10] complete. best={best():.4f}", flush=True)
        break
    print(f"[RESILIENT-P10] crash exit={p.returncode} best={best():.4f}, cooling 30s", flush=True)
    time.sleep(30)
