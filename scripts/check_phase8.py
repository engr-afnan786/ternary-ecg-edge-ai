"""Verify the two Phase 7 binaries and their traced C builds on held-out beats."""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from scripts.build_phase7 import integer, rounded, LIMIT, M, Q, SHAPES
from scripts.phase6_codec import unpack
from scripts.phase5_metrics import metrics

OUT = Path("results/phase8")
P6 = Path("results/phase6")
P7 = Path("results/phase7")
STAGES = ("normalized_input","conv1_relu","max_pool","conv2_relu","global_mean","logits")


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def integer_layers(meta):
    weights = []
    for name, shape in SHAPES.items():
        layer = meta["layers"][name]
        codes = unpack((P6/f"{name}_two_bit.bin").read_bytes(), int(np.prod(shape)), "two_bit")
        weights.append((np.array(codes, dtype=np.int64).reshape(shape),
                        integer(layer["alpha"]*M), np.array([integer(v*Q) for v in layer["bias"]])))
    return weights


def reference(uv, config, layers):
    x = np.clip(rounded((uv.astype(np.int64)-config["mean_uv"])*config["gain_q20"], M), -LIMIT, LIMIT)[None,:]
    result = [x]
    for i,(w,scale,bias) in enumerate(layers[:2]):
        k = w.shape[-1]
        padded = np.pad(x, ((0,0),(k//2,k//2)))
        win = np.lib.stride_tricks.sliding_window_view(padded,k,axis=1)
        stride = 2 if i == 0 else 1
        dot = np.einsum("ctk,ock->ot",win[:,::stride,:],w)
        x = np.clip(rounded(dot*scale,M)+bias[:,None],0,LIMIT)
        result.append(x)
        if i == 0:
            x = x.reshape(8,45,2).max(axis=2)
            result.append(x)
    avg = rounded(x.sum(axis=1),45)
    w,scale,bias = layers[2]
    return result+[avg,np.clip(rounded((w@avg)*scale,M)+bias,-LIMIT,LIMIT)]


def trace_library(packing):
    path = OUT/f"trace{packing}.so"
    subprocess.run(["cc","-std=c11","-Wall","-Wextra","-Werror","-O2","-fPIC","-shared",
                    "-DECG_VERIFY_TRACE",f"-DECG_PACKING={packing}","-Iembedded/include",
                    "-I"+str(P6),"-I"+str(P7),"embedded/src/ecg_integer.c","-o",str(path)],check=True)
    lib = ct.CDLL(str(path.resolve()))
    ptr = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
    lib.ecg_predict_uv.argtypes = [ptr,ptr]
    lib.ecg_predict_uv.restype = ct.c_int
    lib.ecg_read_trace.restype = ct.POINTER(ct.c_int32)
    return lib


def check_trace(lib, uv, expected, production, location):
    scores = np.empty(2,dtype=np.int32)
    label = lib.ecg_predict_uv(uv,scores)
    np.testing.assert_array_equal([*scores,label], production, err_msg=location)
    data = np.ctypeslib.as_array(lib.ecg_read_trace(),shape=(1998,))
    offset = 0
    for name,arr in zip(STAGES,expected,strict=True):
        np.testing.assert_array_equal(data[offset:offset+arr.size],arr.ravel(),err_msg=f"{location} {name}")
        offset += arr.size
    require(offset == len(data), "Trace layout mismatch")


def production_output(packing, text, count):
    buf = subprocess.check_output([str(P7/f"ecg_pack{packing}")],input=text)
    rows = np.fromstring(buf.decode(),sep=" ",dtype=np.int64).reshape(count,3)
    np.testing.assert_array_equal(rows[:,:2].argmax(axis=1),rows[:,2])
    return rows


def main():
    import torch
    from src.models.ecg_ternary import ECGTernary
    OUT.mkdir(parents=True,exist_ok=True)
    dest = OUT/"report.json"
    dest.unlink(missing_ok=True)
    cfg,meta = read(P7/"report.json"),read(P6/"model_metadata.json")
    phase5 = read("results/phase5/report.json")
    manifest = read("results/phase2_dataset.json")
    checkpoint = Path(meta["source_checkpoint"])
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    require(cfg["status"] == phase5["status"] == "PASS" and digest == meta["sha256"] == cfg["checkpoint_sha256"],
            "Phase reports or checkpoint disagree")
    state = torch.load(checkpoint,weights_only=True,map_location="cpu")
    manifest_hash = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    require(state["config"]["manifest_sha256"] == manifest_hash, "Test manifest changed")
    torch.set_num_threads(1)
    model = ECGTernary().eval()
    model.load_state_dict(state["model_state_dict"])
    norm = state["config"]["normalization"]
    layers = integer_layers(meta)
    libs = {p:trace_library(p) for p in (2,5)}
    c_matrix = np.zeros((2,2),dtype=np.int64)
    pt_matrix = np.zeros_like(c_matrix)
    max_error = dict.fromkeys(STAGES,0.)
    n = traced = changes = 0
    abs_sum = max_logit_error = 0.
    for rid,row in sorted(manifest["records"].items()):
        if row["split"] != "test":
            continue
        with np.load(row["file"],allow_pickle=False) as data:
            raw,labels,samples = data["x"],data["y"],data["sample"]
        np.testing.assert_array_equal(np.bincount(labels,minlength=2),row["counts"])
        rounded_uv = np.rint(raw[:,0].astype(np.float64)*1000)
        require(np.all(np.isfinite(rounded_uv) & (np.abs(rounded_uv)<2**31)),f"Invalid input {rid}")
        uv = np.ascontiguousarray(rounded_uv,dtype=np.int32)
        text = "".join(" ".join(map(str,beat))+"\n" for beat in uv).encode()
        scores = production_output(2,text,len(uv))
        np.testing.assert_array_equal(scores,production_output(5,text,len(uv)),err_msg=f"Packing mismatch {rid}")
        x = torch.from_numpy((raw-np.float32(norm["mean_mV"]))/np.float32(norm["std_mV"]))
        with torch.no_grad():
            logits = torch.cat([model(b) for b in x.split(128)]).numpy()
        require(np.isfinite(logits).all(),f"Nonfinite logits {rid}")
        pt = logits.argmax(axis=1)
        np.add.at(c_matrix,(labels,scores[:,2]),1)
        np.add.at(pt_matrix,(labels,pt),1)
        err = np.abs(scores[:,:2]/Q-logits)
        if len(err):
            max_logit_error = max(max_logit_error,float(err.max()))
        abs_sum += float(err.sum())
        n += len(uv)
        changes += int(np.count_nonzero(pt != scores[:,2]))
        for label in (0,1):
            ids = np.flatnonzero(labels==label)
            if not len(ids):
                continue
            for j in np.unique(ids[np.linspace(0,len(ids)-1,min(3,len(ids)),dtype=int)]):
                expected = reference(uv[j],cfg,layers)
                for packing,lib in libs.items():
                    check_trace(lib,uv[j],expected,scores[j],f"{rid}:{samples[j]} pack{packing}")
                with torch.no_grad():
                    v = x[j:j+1]
                    floating = [v]
                    for layer in model.children():
                        v = layer(v)
                        if layer in (model.relu1,model.pool,model.relu2,model.flatten,model.head):
                            floating.append(v)
                for name,a,b in zip(STAGES,expected,floating,strict=True):
                    delta = np.abs(a.ravel()/Q-b.numpy().ravel())
                    max_error[name] = max(max_error[name],float(delta.max()))
                traced += 1
        print(f"Record {rid}: {len(labels)} beats checked",flush=True)
    expected_counts = manifest["splits"]["test"]
    np.testing.assert_array_equal(c_matrix.sum(axis=1),[expected_counts["N"],expected_counts["V"]])
    np.testing.assert_array_equal(pt_matrix,phase5["models"]["ternary"]["test"]["confusion_matrix"],
                                  err_msg="PyTorch did not reproduce Phase 5")
    require(traced > 0 and n>0,"Empty verification set")
    m = metrics(c_matrix)
    report = dict(status="PASS",checkpoint_sha256=digest,test_beats=n,traced_beats=traced,
                  integer_tolerance=0,trace_stages=STAGES,integer_test=m,pytorch_test=metrics(pt_matrix),
                  changed_predictions=changes,agreement=1-changes/n,mean_abs_logit_error=abs_sum/(2*n),
                  max_abs_logit_error=max_logit_error,trace_max_abs_float_difference=max_error)
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False))
    tmp.replace(dest)
    print(f"Exact integer traces PASS | {traced} beats | 6 stages | both formats")
    print(f"Integer C TEST | accuracy {m['accuracy']:.4f} | macro-F1 {m['macro_f1']:.4f} | V recall {m['classes']['V']['recall']:.4f}")
    print("Confusion matrix [N,V]:",m["confusion_matrix"])
    print(f"C vs PyTorch | {changes}/{n} predictions differ | maximum absolute logit error {max_logit_error:.6g}")
    print("Phase 8 PASS |",dest)


if __name__ == "__main__":
    main()
