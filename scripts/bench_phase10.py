"""Linux desktop, one CPU core, batch one. Run from the project root."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
from scripts.phase5_metrics import metrics
from scripts.phase6_codec import unpack

OUT = Path("results/phase10")
C = Path("benchmarks")
NAMES = ["fp32","int8","unpacked","pack2","pack5","cached5"]


def read(path):
    return json.loads(Path(path).read_text())


def floats(values):
    return ",".join(float(v).hex()+"f" for v in values)


def model_and_norm():
    import torch
    from src.models.ecg_fp32 import ECGFP32
    torch.set_num_threads(1)
    data = torch.load("results/phase5/fp32_best.pt",map_location="cpu",weights_only=True)
    model = ECGFP32().eval()
    model.load_state_dict(data["model_state_dict"])
    return model, data["config"]["normalization"]


def calibrate(manifest):
    import torch
    model,norm = model_and_norm()
    maxima = np.zeros(3)
    ids = []
    with torch.inference_mode():
        for rid,row in sorted(manifest["records"].items()):
            if row["split"] != "train":
                continue
            with np.load(row["file"]) as d:
                if not len(d["x"]):
                    continue
                batch = d["x"][np.linspace(0,len(d["x"])-1,min(32,len(d["x"])),dtype=int)]
            ids.append([rid,len(batch)])
            x = torch.from_numpy((batch-np.float32(norm["mean_mV"]))/np.float32(norm["std_mV"]))
            a = model.pool(model.relu1(model.conv1(x)))
            b = model.relu2(model.conv2(a))
            maxima = np.maximum(maxima,[v.abs().max().item() for v in (x,a,b)])
    scales = np.maximum(maxima,1e-9)/127
    lines = [f"static const float s{i}={floats([s])};" for i,s in enumerate(scales)]
    for key in ("mean","std"):
        lines.append(f"static const float {key}_mv={floats([norm[key+'_mV']])};")
    reference_layers=[]
    for i,layer in enumerate((model.conv1,model.conv2,model.head),1):
        wshape=tuple(layer.weight.shape)
        w = layer.weight.detach().numpy().flatten()
        scale = max(float(abs(w).max()),1e-9)/127
        q = np.rint(w/scale).clip(-127,127).astype(np.int8)
        reference_layers.append((q.reshape(wshape).astype(np.int32),np.float32(scale),layer.bias.detach().numpy()))
        lines.extend([f"static const int8_t w{i}[]={{{','.join(map(str,q))}}};",
                      f"static const float sw{i}={floats([scale])}, b{i}[]={{{floats(layer.bias.detach().numpy())}}};"])
    (OUT/"baseline8.h").write_text("\n".join(lines))
    return dict(training_records_and_sample_counts=ids,activation_scales=scales.tolist()), (norm,scales.astype(np.float32),reference_layers)


def verify_int8(raw, outputs, params):
    norm,s,layers=params
    for j in np.linspace(0,len(raw)-1,min(32,len(raw)),dtype=int):
        x=raw[j].astype(np.float32)*np.float32(.001)
        x=np.rint(((x-np.float32(norm["mean_mV"]))/np.float32(norm["std_mV"]))/s[0]).clip(-127,127).astype(np.int32)[None,:]
        for i,(w,sw,bias) in enumerate(layers[:2]):
            k=w.shape[-1]
            windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,((0,0),(k//2,k//2))),k,axis=1)
            if i==0:
                windows=windows[:,::2,:]
            dots=np.einsum("ctk,ock->ot",windows,w)
            x=np.rint((dots.astype(np.float32)*(s[i]*sw)+bias[:,None])/s[i+1]).clip(0,127).astype(np.int32)
            if i==0:
                x=x.reshape(8,45,2).max(axis=2)
        avg=np.rint(x.mean(axis=1)).astype(np.int32)
        w,sw,bias=layers[2]
        logits=(w@avg).astype(np.float32)*(s[2]*sw)+bias
        np.testing.assert_array_equal(np.rint(logits*4096).astype(np.int64),outputs[j,1:3])
        if logits.argmax()!=outputs[j,3]:
            raise ValueError("INT8 reference class mismatch")


def make_test(manifest):
    labels=[]
    with (OUT/"inputs.bin").open("wb") as f:
        for _,row in sorted(manifest["records"].items()):
            if row["split"] != "test":
                continue
            with np.load(row["file"]) as d:
                uv=np.rint(d["x"].astype(np.float64)*1000)
                if not (np.isfinite(uv).all() and (np.abs(uv)<2**31).all()):
                    raise ValueError("Invalid ECG input")
                uv.astype(np.int32).tofile(f)
                labels.extend(d["y"].tolist())
    return np.array(labels)


def fp_worker():
    import torch
    model,norm=model_and_norm()
    uv=np.memmap(OUT/"inputs.bin",dtype=np.int32,mode="c").reshape(-1,1,180)
    beats=torch.from_numpy(uv)
    def infer(x):
        scores=model((x.float().mul_(0.001).sub_(norm["mean_mV"]).div_(norm["std_mV"])))
        return scores.argmax(1).item()
    with torch.inference_mode():
        for _ in range(32):
            infer(beats[:1])
        for i in range(len(beats)):
            beat=beats[i:i+1]
            start=perf_counter_ns()
            label=infer(beat)
            elapsed=perf_counter_ns()-start
            print(elapsed,0,0,label)
    status=Path("/proc/self/status").read_text().splitlines()
    print(next(s for s in status if s.startswith("VmHWM:")).split()[1],file=sys.stderr)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=OUT/"report.json"
    report.unlink(missing_ok=True)
    cpu=min(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{cpu})
    p8,p9=[read(f"results/phase{i}/report.json") for i in (8,9)]
    digest=hashlib.sha256(Path("results/phase5/ternary_best.pt").read_bytes()).hexdigest()
    if p8["status"]!="PASS" or p9["status"]!="PASS" or digest!=p8["checkpoint_sha256"] or digest!=p9["checkpoint_sha256"]:
        raise ValueError("Phase 8/9 checkpoint mismatch")
    manifest=read("results/phase2_dataset.json")
    calibration,int8_params=calibrate(manifest)
    y=make_test(manifest)
    lines=[]
    for name,count in (("conv1",56),("conv2",640),("head",32)):
        data=Path(f"results/phase6/{name}_five_trit.bin").read_bytes()
        digits=unpack(data,count,"five_trit")
        lines.append(f"static const int8_t {name}_packed[]={{{','.join(map(str,digits))}}};")
    (OUT/"byte_weights.h").write_text("\n".join(lines))
    variants={"int8":[],"unpacked":["-DBENCH_UNPACKED"],"pack2":["-DECG_PACKING=2"],
              "pack5":["-DECG_PACKING=5"],"cached5":["-DECG_PACKING=5","-DBENCH_CACHE"]}
    reference=None
    rows=[]
    payload=[3016,728+104+12+12,728+116,182+116,147+116,147+116]
    workspace=[None,180+360+16,2224,2224,2224,2224]
    for i,name in enumerate(NAMES):
        print(f"Benchmarking {name}: {len(y)} single-beat calls",flush=True)
        exe=OUT/name
        if name=="fp32":
            cmd=[sys.executable,"-m","scripts.bench_phase10","--fp-worker"]
            binary_bytes=static=None
        else:
            source=C/("int8.c" if name=="int8" else "ternary.c")
            subprocess.run(["cc","-std=c11","-O3","-Wall","-Wextra","-Werror","-Iembedded/include",
                            "-Iresults/phase6","-Iresults/phase7","-I"+str(OUT),*variants[name],
                            str(C/"runner.c"),str(source),"-lm","-o",str(exe)],check=True)
            cmd=[str(exe)]
            binary_bytes=exe.stat().st_size
            parts=subprocess.check_output(["size",str(exe)],text=True).splitlines()[-1].split()
            static=int(parts[1])+int(parts[2])
        raw=OUT/f"{name}_raw.txt"
        with (OUT/"inputs.bin").open("rb") as inp, raw.open("w") as output:
            proc=subprocess.run(cmd,stdin=inp,stdout=output,stderr=subprocess.PIPE,text=True)
        if proc.returncode:
            raise RuntimeError(proc.stderr)
        rss=int(proc.stderr.strip().splitlines()[-1])
        values=np.loadtxt(raw,dtype=np.int64,ndmin=2)
        if values.shape!=(len(y),4) or not np.isin(values[:,3],[0,1]).all() or min(values[:,0])<=0:
            raise ValueError(f"Invalid output from {name}")
        m=metrics(np.bincount(2*y+values[:,3],minlength=4).reshape(2,2))
        if name=="int8":
            verify_int8(np.memmap(OUT/"inputs.bin",dtype=np.int32,mode="r").reshape(-1,180),values,int8_params)
        if name=="unpacked":
            reference=values[:,1:].copy()
            np.testing.assert_array_equal(m["confusion_matrix"],p8["integer_test"]["confusion_matrix"])
        if name in ("pack2","pack5","cached5"):
            np.testing.assert_array_equal(values[:,1:],reference,err_msg=name)
        us=values[:,0]/1000
        row=dict(name=name,accuracy=m["accuracy"],macro_f1=m["macro_f1"],V_recall=m["classes"]["V"]["recall"],
                 mean_us=float(us.mean()),median_us=float(np.median(us)),p95_us=float(np.percentile(us,95)),
                 serial_inferences_s=1e6/float(us.mean()),parameter_bytes=payload[i],activation_array_bytes=workspace[i],
                 host_elf_bytes=binary_bytes,host_data_bss_bytes=static,process_peak_rss_KiB=rss,
                 decoded_cache_bytes=728 if name=="cached5" else 0,confusion_matrix=m["confusion_matrix"])
        rows.append(row)
        print(f"{name:9s} | accuracy {row['accuracy']:.4f} | median {row['median_us']:.2f} us | P95 {row['p95_us']:.2f} us",flush=True)
    csv_path=OUT/"comparison.csv"
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    result=dict(status="PASS",cpu_affinity=cpu,host=platform.platform(),compiler=subprocess.check_output(["cc","--version"],text=True).splitlines()[0],
                samples=len(y),warmup_calls=32,seeded_order="Phase 2 record and annotation order",batch_size=1,checkpoint_sha256=digest,
                int8_calibration=calibration,variants=rows,emulated_firmware=p9["builds"],energy_per_inference=None,
                notes="Inputs native int32 microvolts. Timed normalization through class decision. C uses CLOCK_MONOTONIC; Python uses perf_counter_ns. RSS is process-wide. C activation arrays exclude caller input, outputs, scalar locals and compiler stack overhead. PyTorch activation memory not profiled. FP32 parameter totals exclude input normalization; INT8 includes three activation scales. Packed-cache init excluded. No physical board measurements.")
    tmp=report.with_suffix(".tmp")
    tmp.write_text(json.dumps(result,indent=2,allow_nan=False))
    tmp.replace(report)
    print("Phase 10 PASS |",report)


if __name__=="__main__":
    if sys.argv[1:]==["--fp-worker"]:
        fp_worker()
    else:
        main()
