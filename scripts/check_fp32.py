"""Run in Phase 3 from the project root with python -m scripts.check_fp32."""
import json
import platform
from pathlib import Path

import numpy as np
import torch
from src.models.ecg_fp32 import ECGFP32

REPORT = Path("results/phase3_fp32.json")
SHAPES = [(8, 90), (8, 90), (8, 45), (16, 45), (16, 45), (16, 1), (16,), (2,)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def training_inputs(report):
    """Pool per-record population variances in float64 and select a small two-class batch."""
    count, mean, m2 = 0, 0., 0.
    examples = [[], []]
    totals = np.zeros(2, dtype=np.int64)
    for rid, record in report["records"].items():
        if record["split"] != "train":
            continue
        path = Path("datasets/processed/mitdb_nv_v1/train") / f"{rid}.npz"
        require(path == Path(record["file"]), f"Unexpected training path for {rid}")
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"], data["y"]
        require(x.shape == (len(y), 1, 180) and x.dtype == np.float32, f"Bad tensor in {rid}")
        require(y.ndim == 1 and np.isin(y, [0, 1]).all() and np.isfinite(x).all(), f"Bad values in {rid}")
        counts = np.bincount(y, minlength=2)
        require(counts.tolist() == record["counts"], f"Count mismatch in {rid}")
        totals += counts
        if not x.size:
            continue
        n = x.size
        delta = float(x.mean(dtype=np.float64)) - mean
        m2 += n * float(x.var(dtype=np.float64)) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        for label in (0, 1):
            ids = np.flatnonzero(y == label)[:8-len(examples[label])]
            examples[label].extend(x[ids].copy())
    expected = report["splits"]["train"]
    require(totals.tolist() == [expected["N"], expected["V"]], "Training totals differ")
    require(all(len(group) == 8 for group in examples), "Need eight training examples per class")
    std = (m2 / count)**0.5
    require(np.isfinite(std) and std > 0, "Invalid training standard deviation")
    batch = np.stack(examples[0] + examples[1])
    values = (batch - np.float32(mean)) / np.float32(std)
    return torch.from_numpy(values), torch.arange(2).repeat_interleave(8), mean, std, totals


def check():
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.unlink(missing_ok=True)
    phase2 = json.loads(Path("results/phase2_dataset.json").read_text())
    require(phase2["status"] == "PASS" and phase2["labels"] == {"N": 0, "V": 1}, "Check Phase 2 report")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    x, y, mean, std, totals = training_inputs(phase2)
    model = ECGFP32().float()
    params = sum(p.numel() for p in model.parameters())
    require(params == 754, "Unexpected model size")
    with torch.no_grad():
        features = x.clone()
        for (name, layer), shape in zip(model.named_children(), SHAPES, strict=True):
            features = layer(features)
            require(tuple(features.shape) == (16, *shape), f"Unexpected shape after {name}")
            require(torch.isfinite(features).all().item(), f"Nonfinite values after {name}")
        logits = model(x)
        torch.testing.assert_close(logits, features)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(x), y)
    require(torch.isfinite(loss).item(), "Nonfinite loss")
    loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()), "Invalid gradient")
    before = model.head.weight.detach().clone()
    optimizer.step()
    require(not torch.equal(before, model.head.weight), "Optimizer did not update output weights")
    require(torch.isfinite(model(x)).all().item(), "Nonfinite logits after update")
    report = dict(status="PASS", python=platform.python_version(), torch=str(torch.__version__),
                  train_counts=totals.tolist(), normalization=dict(mean_mV=mean, std_mV=std, fitted_on="train"),
                  parameters=params, fp32_parameter_bytes=params*4, seed=42, threads=1,
                  batch_shape=list(x.shape), logits_shape=list(logits.shape), smoke_loss=loss.item())
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(REPORT)
    print(f"Training beats: N={totals[0]}, V={totals[1]}")
    print(f"Parameters: {params} | FP32 parameter bytes: {params*4}")
    print("Forward, backward and optimizer update PASS")
    print("Phase 3 PASS |", REPORT)


if __name__ == "__main__":
    check()
