"""Build and check standalone integer ECG inference from Phase 6 artifacts."""
from pathlib import Path
import json
import math
import subprocess
import numpy as np
from scripts.phase6_codec import unpack

SRC = Path("results/phase6")
OUT = Path("results/phase7")
Q, M, LIMIT = 4096, 1 << 20, 1 << 20
SHAPES = {"conv1": (8, 1, 7), "conv2": (16, 8, 5), "head": (2, 16)}


def read(path):
    return json.loads(path.read_text())


def integer(value):
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def rounded(x, d):
    return np.sign(x) * ((np.abs(x) + d//2)//d)


def config(meta):
    norm = meta["normalization"]
    center = integer(norm["mean_mV"] * 1000)
    gain = integer(Q * M / (1000 * norm["std_mV"]))
    if not (-2**31 < center < 2**31 and 0 < gain < 2**31):
        raise ValueError("Input normalization outside verified integer bounds")
    lines = [f"#define ECG_MEAN_UV ({center})", f"#define ECG_INPUT_GAIN {gain}",
             f"#define ECG_LIMIT {LIMIT}", f"#define ECG_ONE20 {M}"]
    layers = []
    for name, shape in SHAPES.items():
        layer = meta["layers"][name]
        if tuple(layer["shape"]) != shape:
            raise ValueError(f"Unsupported {name} shape")
        n = math.prod(shape)
        a = unpack((SRC/f"{name}_two_bit.bin").read_bytes(), n, "two_bit")
        b = unpack((SRC/f"{name}_five_trit.bin").read_bytes(), n, "five_trit")
        if a != b:
            raise ValueError(f"Different weights in {name}")
        scale = integer(layer["alpha"] * M)
        bias = [integer(v * Q) for v in layer["bias"]]
        if len(bias) != shape[0] or not (0 <= scale < 2**31 and max(map(abs,bias)) < 2**31):
            raise ValueError(f"{name} exceeds supported coefficient range")
        lines += [f"static const int32_t {name}_scale = {scale};",
                  f"static const int32_t {name}_bias_q[] = {{{', '.join(map(str,bias))}}};"]
        layers.append((np.array(a,dtype=np.int64).reshape(shape), scale, np.array(bias,dtype=np.int64)))
    (OUT/"integer_config.h").write_text("#include <stdint.h>\n"+"\n".join(lines)+"\n")
    return center, gain, layers


def reference(beat, center, gain, layers):
    x = np.clip(rounded((beat.astype(np.int64)-center)*gain,M),-LIMIT,LIMIT)[None,:]
    for i, (w,scale,bias) in enumerate(layers[:2]):
        k = w.shape[-1]
        windows = np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(k//2,k//2))),k,axis=1)
        if i == 0:
            windows = windows[:,::2,:]
        dot = np.einsum("ctk,ock->ot",windows,w)
        x = np.clip(rounded(dot*scale,M)+bias[:,None],0,LIMIT)
        if i == 0:
            x = x.reshape(8,45,2).max(axis=2)
    w,scale,bias = layers[2]
    score = np.clip(rounded((w @ rounded(x.sum(axis=1),45))*scale,M)+bias,-LIMIT,LIMIT)
    return [*map(int,score),int(score.argmax())]


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    report = OUT/"report.json"
    report.unlink(missing_ok=True)
    if read(SRC/"model_statistics.json")["status"] != "PASS":
        raise ValueError("Phase 6 PASS required")
    meta = read(SRC/"model_metadata.json")
    center,gain,layers = config(meta)
    beats, names = [], []
    for item in meta["golden_validation_sources"]:
        rid = item["record"]
        with np.load(f"datasets/processed/mitdb_nv_v1/validation/{rid}.npz") as data:
            idx = np.flatnonzero((data["sample"] == item["sample"]) & (data["y"] == item["label"]))
            if len(idx) != 1:
                raise ValueError(f"Validation beat mismatch: {item}")
            uv = np.rint(data["x"][idx[0],0].astype(np.float64)*1000)
            if not np.all(np.isfinite(uv) & (np.abs(uv) < 2**31)):
                raise ValueError("Invalid ECG microvolt input")
            beats.append(uv.astype(np.int64))
            names.append(f"record {rid} sample {item['sample']}")
    for label, beat in [("zero",np.zeros(180,dtype=np.int64)),
                        ("signed ramp",np.arange(-90,90)*25),
                        ("alternating int32 limits",np.tile([-2**31,2**31-1],90))]:
        names.append(label)
        beats.append(beat)
    expected = [reference(b,center,gain,layers) for b in beats]
    inputs = "".join(" ".join(map(str,b))+"\n" for b in beats)
    (OUT/"example_beat_uv.txt").write_text(inputs.splitlines()[0]+"\n")
    for packing in (2,5):
        exe = OUT/f"ecg_pack{packing}"
        subprocess.run(["cc","-std=c11","-O2","-Wall","-Wextra","-Werror",f"-DECG_PACKING={packing}",
                        "-Iembedded/include","-I"+str(SRC),"-I"+str(OUT),"embedded/src/ecg_integer.c",
                        "embedded/src/ecg_cli.c","-o",str(exe)],check=True)
        rows = subprocess.check_output([str(exe)],input=inputs,text=True).splitlines()
        actual = [list(map(int,row.split())) for row in rows]
        if actual != expected:
            raise ValueError(f"C/Python integer score mismatch for packing {packing}: {actual} vs {expected}")
        print(f"Packing {packing}: {len(beats)} integer reference comparisons PASS")
    tmp = report.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(status="PASS", checkpoint_sha256=meta["sha256"], input_unit="microvolt",
                                   score_divisor=Q, activation_limit=LIMIT, mean_uv=center, gain_q20=gain,
                                   inputs=names, expected_scores_and_class=expected),indent=2))
    tmp.replace(report)
    print("Phase 7 PASS |",report)


if __name__ == "__main__":
    build()
