import torch, json
from pathlib import Path

h = Path('checkpoints/run_full/history.json')
if h.exists():
    data = json.load(open(h))
    print('Epoch history:')
    for ep in data:
        print(f"  Ep {ep['ep']:03d}: train={ep['tr']:.4f}  val={ep['va']:.4f}")
else:
    print('No history.json found')

ckpt = torch.load('checkpoints/run_full/checkpoint_best.pt', map_location='cpu')
cfg = ckpt.get('config', {})
print('\nTraining config used:')
for k, v in cfg.items():
    print(f"  {k}: {v}")