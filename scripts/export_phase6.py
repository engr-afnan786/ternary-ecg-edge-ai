"""Export the selected Phase 5 checkpoint to two standalone C representations."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import subprocess

import numpy as np
import torch

from scripts.phase6_codec import FORMATS, pack, unpack, self_test
from src.models.ecg_fp32 import ECGFP32
from src.models.ecg_ternary import ECGTernary
from src.quantization.ternary import components

OUT = Path("results/phase6")
CKPT = Path("results/phase5/ternary_best.pt")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def floats(values):
    return ", ".join(float(v).hex() + "f" for v in values)


def header(path, lines):
    guard = "ECG_" + path.name.upper().replace(".", "_")
    path.write_text("\n".join([f"#ifndef {guard}", f"#define {guard}", "#include <stdint.h>", *lines, "#endif", ""]))


def golden_inputs(manifest):
    beats, sources = {}, {}
    for rid, row in sorted(manifest["records"].items()):
        if row["split"] != "validation":
            continue
        with np.load(row["file"], allow_pickle=False) as data:
            for label in (0, 1):
                ids = np.flatnonzero(data["y"] == label)
                if label not in beats and len(ids):
                    i = ids[0]
                    beats[label] = data["x"][i].copy()
                    sources[label] = dict(record=rid, sample=int(data["sample"][i]), label=label)
        if len(beats) == 2:
            break
    return torch.from_numpy(np.stack([beats[0], beats[1]])), [sources[0], sources[1]]


@torch.no_grad()
def export(folder):
    report = read("results/phase5/report.json")
    checkpoint = torch.load(CKPT, weights_only=True, map_location="cpu")
    cfg = checkpoint["config"]
    selection = report["models"]["ternary"]
    require(report["status"] == "PASS" and checkpoint["model"] == "ternary" and
            cfg == report["config"] and checkpoint["epoch"] == selection["best_epoch"] and
            checkpoint["validation_macro_f1"] == selection["validation_macro_f1"], "Phase 5 checkpoint mismatch")
    manifest = read("results/phase2_dataset.json")
    require(hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() == cfg["manifest_sha256"],
            "Dataset manifest mismatch")
    torch.set_num_threads(1)
    model = ECGTernary().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    norm = cfg["normalization"]
    raw, sources = golden_inputs(manifest)
    mean, std = np.float32(norm["mean_mV"]), np.float32(norm["std_mV"])
    x = (raw - float(mean)) / float(std)
    logits = model(x)
    require(torch.isfinite(logits).all(), "Nonfinite reference scores")
    config = ["#define ECG_INPUT_LENGTH 180", "#define ECG_CLASS_COUNT 2",
              f"static const float ecg_mean_mv = {floats([mean])};",
              f"static const float ecg_std_mv = {floats([std])};"]
    weight_lines = ["/* Select 2 for two-bit or 5 for five-trit before including. */",
                    "#ifndef ECG_PACKING", "#define ECG_PACKING 5", "#endif",
                    "#if ECG_PACKING != 2 && ECG_PACKING != 5", '#error "Invalid ECG_PACKING"', "#endif"]
    layers, sizes = {}, dict.fromkeys(FORMATS, 0)
    refs = {fmt: ECGFP32().eval() for fmt in FORMATS}
    for name in model.weight_layers:
        layer = getattr(model, name)
        q, alpha, delta = components(layer.parametrizations.weight.original)
        require(torch.isfinite(layer.weight).all() and torch.isfinite(layer.bias).all(), "Nonfinite parameters")
        codes = q.flatten().tolist()
        counts = [codes.count(i) for i in (-1, 0, 1)]
        recorded = report["ternary_layers"][name]
        require(counts == recorded["counts_minus_zero_plus"] and alpha.item() == recorded["alpha"] and
                delta.item() == recorded["threshold"], f"Checkpoint statistics differ for {name}")
        config.extend([f"#define ECG_{name.upper()}_COUNT {len(codes)}",
                       f"static const float {name}_alpha = {floats([alpha])};",
                       f"static const float {name}_bias[] = {{{floats(layer.bias)}}};"])
        layer_bytes = {}
        for fmt in FORMATS:
            data = pack(codes, fmt)
            filename = f"{name}_{fmt}.bin"
            (folder / filename).write_bytes(data)
            decoded = unpack((folder / filename).read_bytes(), len(codes), fmt)
            require(decoded == codes, f"Symbol mismatch in {filename}")
            reference = getattr(refs[fmt], name)
            reference.weight.copy_(torch.tensor(decoded).reshape_as(q) * alpha)
            reference.bias.copy_(layer.bias)
            require(torch.equal(reference.weight, layer.weight), "Effective weight mismatch")
            sizes[fmt] += len(data)
            layer_bytes[fmt] = len(data)
            weight_lines += ["#if ECG_PACKING == 2" if fmt == "two_bit" else "#else",
                             f"static const uint8_t {name}_packed[] = {{{', '.join(map(str, data))}}};"]
        weight_lines.append("#endif")
        layers[name] = dict(shape=list(q.shape), count=len(codes), alpha=alpha.item(), threshold=delta.item(),
                            bias=layer.bias.tolist(), counts_minus_zero_plus=counts, bytes=layer_bytes)
    for ref in refs.values():
        torch.testing.assert_close(ref(x), logits, atol=0, rtol=0)
    header(folder / "model_config.h", config)
    header(folder / "ternary_weights.h", weight_lines)
    header(folder / "golden_test_vectors.h", [
        f"static const float ecg_golden_raw_mv[2][180] = {{{', '.join('{'+floats(r.flatten())+'}' for r in raw)}}};",
        f"static const float ecg_golden_logits[2][2] = {{{', '.join('{'+floats(r)+'}' for r in logits)}}};",
        f"static const int ecg_golden_prediction[2] = {{{', '.join(map(str, logits.argmax(1).tolist()))}}};"])
    # Compile the generated headers twice and compare the actual bytes in the C arrays.
    probe = folder / "probe.c"
    probe.write_text('#include "model_config.h"\n#include "ternary_weights.h"\n#include "golden_test_vectors.h"\n#include <stdio.h>\nint main(void){\n' +
                     ''.join(f'fwrite({n}_packed,1,sizeof({n}_packed),stdout);\n' for n in layers) + 'return ferror(stdout);}\n')
    for fmt, selector in [("two_bit", 2), ("five_trit", 5)]:
        exe = folder / "probe"
        subprocess.run(["cc", "-std=c11", f"-DECG_PACKING={selector}", str(probe), "-o", str(exe)], check=True)
        require(subprocess.check_output([str(exe.resolve())]) == b"".join(
            (folder / f"{name}_{fmt}.bin").read_bytes() for name in layers), "C header byte mismatch")
    exe.unlink()
    probe.unlink()
    meta = dict(source_checkpoint=str(CKPT), sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest(),
                epoch=checkpoint["epoch"], layers=layers, labels=cfg["labels"], normalization=norm,
                layout="C order: convolution [out,in,kernel], dense [out,in]", digit="code+1, least significant first, unused high digits zero",
                topology=["input 1x180", "conv1 stride 2 pad 3 -> 8x90", "ReLU", "maxpool 2 stride 2 -> 8x45",
                          "conv2 stride 1 pad 2 -> 16x45", "ReLU", "global mean -> 16", "head -> 2"],
                golden_validation_sources=sources, activations="float32")
    (folder / "model_metadata.json").write_text(json.dumps(meta, indent=2, allow_nan=False))
    stats = dict(status="PASS", symbols=sum(l["count"] for l in layers.values()), weight_bytes=sizes,
                 bias_bytes=104, scale_bytes=12, normalization_bytes=8,
                 parameter_bytes={k:v+116 for k,v in sizes.items()})
    require(stats["symbols"] == 728 and sizes == {"two_bit":182,"five_trit":147}, "Unexpected model layout")
    (folder / "model_statistics.json").write_text(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "model_statistics.json").unlink(missing_ok=True)
    self_test()
    with TemporaryDirectory(dir=OUT) as tmp:
        folder = Path(tmp)
        stats = export(folder)
        for path in sorted(folder.iterdir(), key=lambda p: p.name == "model_statistics.json"):
            path.replace(OUT / path.name)
    print("728 weights verified in both formats | 2-bit: 182 bytes | five-trit: 147 bytes")
    print("Compiled headers and validation logits PASS")
    print("Phase 6 PASS | results/phase6/model_statistics.json")
