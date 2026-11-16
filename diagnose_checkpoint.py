import torch
from pathlib import Path

ckpt = torch.load('checkpoints/run_full/checkpoint_best.pt', map_location='cpu')
sd = ckpt['model']

print("=== Gate layer weights (should NOT be all-zero if training worked) ===")
for k, v in sd.items():
    if 'gate' in k:
        print(f"  {k}: shape={list(v.shape)}  mean={v.float().mean().item():.6f}  std={v.float().std().item():.6f}")

print("\n=== Projection layers (proj_a = EfficientNet, proj_b = handcrafted) ===")
for k, v in sd.items():
    if 'proj_a.0' in k or 'proj_b.0' in k:
        print(f"  {k}: shape={list(v.shape)}  mean={v.float().mean().item():.6f}  std={v.float().std().item():.6f}")

print("\n=== Input BatchNorm (handcrafted branch) ===")
for k, v in sd.items():
    if 'input_bn' in k:
        print(f"  {k}: shape={list(v.shape)}  mean={v.float().mean().item():.6f}")

print("\n=== EfficientNet first conv ===")
for k, v in sd.items():
    if 'conv_stem' in k and 'weight' in k:
        print(f"  {k}: shape={list(v.shape)}  mean={v.float().mean().item():.6f}  std={v.float().std().item():.6f}")
        break

print("\n=== Classifier weights ===")
for k, v in sd.items():
    if 'classifier' in k and 'weight' in k:
        print(f"  {k}: shape={list(v.shape)}  mean={v.float().mean().item():.6f}  std={v.float().std().item():.6f}")

# Run a quick forward pass to check gate behavior
print("\n=== Simulated gate behavior ===")
try:
    import sys
    sys.path.insert(0, '.')
    from models.detector import build_detector
    model = build_detector(ckpt['config'])
    model.load_state_dict(sd)
    model.eval()

    import torch
    with torch.no_grad():
        # Test with random inputs
        di = torch.rand(16, 3, 224, 224)
        hc = torch.rand(16, 9369)
        logits, gates = model(di, hc, return_gates=True)
        alphas = gates[:, 0].numpy()
        print(f"  Alpha (EfficientNet weight): min={alphas.min():.4f}  max={alphas.max():.4f}  mean={alphas.mean():.4f}")
        print(f"  Beta  (Handcrafted weight):  min={gates[:,1].min().item():.4f}  max={gates[:,1].max().item():.4f}  mean={gates[:,1].mean().item():.4f}")
        probs = torch.sigmoid(logits)
        print(f"  Output probs: min={probs.min().item():.4f}  max={probs.max().item():.4f}  mean={probs.mean().item():.4f}")
        if probs.max() - probs.min() < 0.05:
            print("  WARNING: Model outputs near-constant probability — not discriminating")
        else:
            print("  OK: Model outputs vary across inputs")
except Exception as e:
    print(f"  Forward pass failed: {e}")