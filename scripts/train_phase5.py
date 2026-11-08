"""Run from the project root with python -u -m scripts.train_phase5."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from torch.nn import functional as F

from scripts.phase5_metrics import metrics
from src.models.ecg_fp32 import ECGFP32
from src.models.ecg_ternary import ECGTernary
from src.quantization.ternary import components

OUT = Path("results/phase5")
SEED, EPOCHS, BATCH, LR = 42, 20, 128, 0.001
MODELS = {"fp32": ECGFP32, "ternary": ECGTernary}


def save(path, value):
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def load_split(manifest, split, norm):
    xs, ys = [], []
    for rid, row in sorted(manifest["records"].items()):
        if row["split"] != split:
            continue
        path = Path("datasets/processed/mitdb_nv_v1") / split / f"{rid}.npz"
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"], data["y"]
        if x.shape != (len(y), 1, 180) or x.dtype != np.float32 or y.dtype != np.int64:
            raise ValueError(f"Unexpected data layout in {path}")
        if not np.isin(y, (0, 1)).all() or np.bincount(y, minlength=2).tolist() != row["counts"]:
            raise ValueError(f"Label mismatch in {path}")
        x -= np.float32(norm["mean_mV"])
        x /= np.float32(norm["std_mV"])
        if not np.isfinite(x).all():
            raise ValueError(f"Nonfinite input in {path}")
        xs.append(x)
        ys.append(y)
    labels = np.concatenate(ys)
    counts = manifest["splits"][split]
    if np.bincount(labels, minlength=2).tolist() != [counts["N"], counts["V"]]:
        raise ValueError(f"Split totals differ for {split}")
    return torch.from_numpy(np.concatenate(xs)), torch.from_numpy(labels)


def run_batches(model, data, order, weights, optimizer=None):
    x, y = data
    matrix = torch.zeros(4, dtype=torch.int64)
    loss_sum = weight_sum = 0.
    model.train(optimizer is not None)
    with torch.set_grad_enabled(optimizer is not None):
        for idx in order.split(BATCH):
            target = y[idx]
            scores = model(x[idx])
            loss = F.cross_entropy(scores, target, weight=weights, reduction="sum")
            mass = weights[target].sum()
            if not torch.isfinite(loss).item():
                raise ValueError("Nonfinite loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                (loss / mass).backward()
                optimizer.step()
            loss_sum += loss.item()
            weight_sum += mass.item()
            matrix += torch.bincount(2 * target + scores.detach().argmax(1), minlength=4)
    result = metrics(matrix.reshape(2, 2).numpy())
    result["weighted_loss"] = loss_sum / weight_sum
    return result


def train(name, train_data, val_data, weights, config):
    torch.manual_seed(SEED)
    model = MODELS[name]()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    latest = OUT / f"{name}_resume.pt"
    history, best, best_score, best_epoch = [], None, -1., 0
    if latest.exists():
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if config != state["config"]:
            raise ValueError(f"Settings differ from {latest}. Move results/phase5 aside for a fresh run.")
        model.load_state_dict(state["current"])
        optimizer.load_state_dict(state["optimizer"])
        history, best, best_score, best_epoch = (state[k] for k in ("history", "best", "best_score", "best_epoch"))
        print(f"{name}: recovered {len(history)} completed epochs")
    for epoch in range(len(history) + 1, EPOCHS + 1):
        order = torch.randperm(len(train_data[1]), generator=torch.Generator().manual_seed(SEED + epoch))
        tr = run_batches(model, train_data, order, weights, optimizer)
        val = run_batches(model, val_data, torch.arange(len(val_data[1])), weights)
        if val["macro_f1"] > best_score:
            best_score, best_epoch = val["macro_f1"], epoch
            best = deepcopy(model.state_dict())
        history.append(dict(epoch=epoch, train=tr, validation=val))
        save(latest, dict(config=config, current=model.state_dict(), optimizer=optimizer.state_dict(),
                          history=history, best=best, best_score=best_score, best_epoch=best_epoch))
        print(f"{name} epoch {epoch:02d}/{EPOCHS}: train loss {tr['weighted_loss']:.4f}, validation macro-F1 {val['macro_f1']:.4f}", flush=True)
    path = OUT / f"{name}_best.pt"
    save(path, dict(model=name, model_state_dict=best, epoch=best_epoch,
                    validation_macro_f1=best_score, config=config))
    model.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
    return model, dict(best_epoch=best_epoch, validation_macro_f1=best_score, history=history,
                       checkpoint=str(path), checkpoint_bytes=path.stat().st_size)


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for row, (name, result) in enumerate(results.items()):
        hist = result["history"]
        for ax, metric in zip(axes[row], ("weighted_loss", "macro_f1")):
            for split in ("train", "validation"):
                ax.plot([h["epoch"] for h in hist], [h[split][metric] for h in hist], label=split)
            ax.axvline(result["best_epoch"], linestyle=":", color="gray")
            ax.set(title=f"{name} {metric}", xlabel="Epoch")
            ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / "report.json"
    report.unlink(missing_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    phases = [json.loads(Path(f"results/{name}.json").read_text()) for name in
              ("phase2_dataset", "phase3_fp32", "phase4_ternary")]
    if any(p["status"] != "PASS" for p in phases):
        raise ValueError("Phase 2, 3 and 4 PASS reports required")
    manifest = phases[0]
    norm = phases[1]["normalization"]
    train_data = load_split(manifest, "train", norm)
    val_data = load_split(manifest, "validation", norm)
    counts = torch.bincount(train_data[1]).float()
    weights = counts.sum() / (2 * counts)
    config = dict(seed=SEED, epochs=EPOCHS, batch=BATCH, lr=LR, optimizer="Adam",
                  weights=weights.tolist(), normalization=norm, labels=manifest["labels"],
                  torch=str(torch.__version__), python=platform.python_version(), threads=1,
                  manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    models, results = {}, {}
    for name in MODELS:
        models[name], results[name] = train(name, train_data, val_data, weights, config)
    del train_data, val_data
    test_data = load_split(manifest, "test", norm)
    for name, model in models.items():
        test = run_batches(model, test_data, torch.arange(len(test_data[1])), weights)
        results[name]["test"] = test
        print(f"{name} TEST | accuracy {test['accuracy']:.4f} | macro-F1 {test['macro_f1']:.4f} | V recall {test['classes']['V']['recall']:.4f}")
        print(f"Confusion matrix [N,V]: {test['confusion_matrix']}")
    layers = {}
    for name in models["ternary"].weight_layers:
        weight = getattr(models["ternary"], name).parametrizations.weight.original
        q, alpha, delta = components(weight)
        counts = torch.bincount(q.flatten().long()+1, minlength=3)
        layers[name] = dict(counts_minus_zero_plus=counts.tolist(), zero_percent=100*counts[1].item()/q.numel(),
                            alpha=alpha.item(), threshold=delta.item())
    plot(results)
    tmp = report.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(status="PASS", config=config, models=results, ternary_layers=layers),
                              indent=2, allow_nan=False))
    tmp.replace(report)
    print("Phase 5 PASS |", report)


if __name__ == "__main__":
    main()
