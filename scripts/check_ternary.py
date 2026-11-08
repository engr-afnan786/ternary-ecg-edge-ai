"""Run with python -m scripts.check_ternary from the project root."""
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from scripts.check_fp32 import training_inputs, require
from src.models.ecg_fp32 import ECGFP32
from src.models.ecg_ternary import ECGTernary
from src.quantization.ternary import TernaryWeight, components


def test_quantizer():
    # mean(abs(w)) = 1.0. Both threshold ties must map to zero.
    w = torch.tensor([-1.3, -0.7, 0.7, 1.3], dtype=torch.float64, requires_grad=True)
    q, alpha, delta = components(w)
    torch.testing.assert_close(q, torch.tensor([-1, 0, 0, 1], dtype=torch.int8))
    torch.testing.assert_close(delta, w.new_tensor(0.7))
    torch.testing.assert_close(alpha, w.new_tensor(1.3))
    effective = TernaryWeight()(w)
    torch.testing.assert_close(effective, w.new_tensor([-1.3, 0, 0, 1.3]))
    effective.backward(w.new_tensor([2, 3, 5, 7]))
    torch.testing.assert_close(w.grad, w.new_tensor([0, 3, 5, 0]))
    for values, expected_grad in [([0., 0.], [1., 1.]), ([-1., 1.], [1., 1.])]:
        probe = torch.tensor(values, requires_grad=True)
        out = TernaryWeight()(probe)
        torch.testing.assert_close(out, probe.detach())
        out.sum().backward()
        torch.testing.assert_close(probe.grad, torch.tensor(expected_grad))


def check():
    torch.manual_seed(42)
    torch.set_num_threads(1)
    dest = Path("results/phase4_ternary.json")
    dest.unlink(missing_ok=True)
    test_quantizer()
    print("Threshold, scale and STE checks PASS")
    p3 = json.loads(Path("results/phase3_fp32.json").read_text())
    require(p3["status"] == "PASS", "Phase 3 PASS required")
    p2 = json.loads(Path("results/phase2_dataset.json").read_text())
    require(p2["status"] == "PASS", "Phase 2 PASS required")
    x, y, mean, std, _ = training_inputs(p2)
    norm = p3["normalization"]
    require((mean, std) == (norm["mean_mV"], norm["std_mV"]), "Training normalization differs from Phase 3")
    model = ECGTernary()
    reference = ECGFP32()
    summary = {}
    before = {}
    with torch.no_grad():
        for name in model.weight_layers:
            layer = getattr(model, name)
            latent = layer.parametrizations.weight.original
            before[name] = latent.clone()
            # Independent scalar reference, including scale and strict tie policy.
            values = latent.flatten().tolist()
            delta = 0.7 * sum(abs(v) for v in values) / len(values)
            codes = [int(v > delta) - int(v < -delta) for v in values]
            active = [abs(v) for v, code in zip(values, codes) if code]
            alpha = sum(active) / max(1, len(active))
            q = torch.tensor(codes, dtype=torch.int8).reshape_as(latent)
            actual, _, _ = components(latent)
            torch.testing.assert_close(actual, q)
            ref_layer = getattr(reference, name)
            ref_layer.weight.copy_(q.float() * alpha)
            ref_layer.bias.copy_(layer.bias)
            summary[name] = dict(negative=codes.count(-1), zero=codes.count(0), positive=codes.count(1),
                                 zero_percent=100 * codes.count(0)/len(codes), alpha=alpha)
    logits = model(x)
    require(logits.shape == (16, 2) and torch.isfinite(logits).all().item(), "Invalid logits")
    with torch.no_grad():
        torch.testing.assert_close(logits, reference(x), atol=1e-5, rtol=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
            "Missing or nonfinite gradient")
    optimizer.step()
    for name in model.weight_layers:
        latent = getattr(model, name).parametrizations.weight.original
        require(not torch.equal(before[name], latent), f"No latent weight update in {name}")
    require(torch.isfinite(model(x)).all().item(), "Invalid post-update logits")
    count = sum(v["negative"] + v["zero"] + v["positive"] for v in summary.values())
    require(count == 728 and sum(p.numel() for p in model.parameters()) == 754, "Topology mismatch")
    report = dict(status="PASS", seed=42, torch=str(torch.__version__), weight_symbols=count,
                  fp32_biases=26, initial_layers=summary, smoke_loss=loss.item(), batch_shape=list(x.shape))
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(dest)
    print("Ternary weights: 728 | FP32 biases: 26")
    print("Real ECG forward, reference comparison and optimizer update PASS")
    print("Phase 4 PASS |", dest)


if __name__ == "__main__":
    check()
