"""Audit existing evidence in the ECG project root; no model training."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

R = Path("results")
OUT = R/"phase11"
NAMES = ("fp32","int8","unpacked","pack2","pack5","cached5")
SHAPES = {"conv1":(8,1,7),"conv2":(16,8,5),"head":(2,16)}


def read(path):
    return json.loads(Path(path).read_text())


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evidence():
    files = ["phase1_environment.json","phase2_dataset.json","phase3_fp32.json","phase4_ternary.json",
             "phase5/report.json","phase6/model_statistics.json"]+[f"phase{i}/report.json" for i in (7,8,9,10)]
    for name in files:
        check(read(R/name)["status"] == "PASS",f"Missing PASS in {name}")
    identity = digest(R/"phase5/ternary_best.pt")
    check(read(R/"phase6/model_metadata.json")["sha256"] == identity,"Export checkpoint identity")
    for i in (7,8,9,10):
        check(read(R/f"phase{i}/report.json")["checkpoint_sha256"]==identity,f"Phase {i} checkpoint identity")
    return f"10 phase statuses checked. Ternary checkpoint SHA-256 {identity}."


def split():
    m=read(R/"phase2_dataset.json")
    groups={s:set() for s in ("train","validation","test")}
    counts={s:np.zeros(2,dtype=int) for s in groups}
    for rid,row in m["records"].items():
        s=row["split"]
        groups[s].add("201" if rid=="202" else rid)
        path=Path("datasets/processed/mitdb_nv_v1")/s/f"{rid}.npz"
        check(Path(row["file"])==path,f"Unexpected record path {rid}")
        with np.load(path) as d:
            x,y,t=d["x"],d["y"],d["sample"]
            check(x.shape==(len(y),1,180) and np.isfinite(x).all() and np.isin(y,[0,1]).all(), f"Bad windows {rid}")
            check(t.shape==y.shape and np.all(np.diff(t)>0),f"Duplicate or unordered beats {rid}")
            c=np.bincount(y,minlength=2)
        np.testing.assert_array_equal(c,row["counts"])
        counts[s]+=c
    for s,patients in groups.items():
        info=m["splits"][s]
        check(len(patients)==info["patients"],f"{s} patient count")
        np.testing.assert_array_equal(counts[s],[info["N"],info["V"]])
    check(sum(map(len,groups.values()))==len(set.union(*groups.values()))==43,"Patient overlap or missing patients")
    cfg=read(R/"phase10/report.json")["int8_calibration"]["training_records_and_sample_counts"]
    check(all(m["records"][rid]["split"]=="train" and n>0 for rid,n in cfg),"Calibration leakage")
    return "43 disjoint patients. Verified window shapes, labels, sample indices and training-only INT8 calibration records."


def quantization():
    import torch
    from src.quantization.ternary import TernaryWeight
    from src.models.ecg_ternary import ECGTernary
    torch.set_num_threads(1)
    model=ECGTernary()
    state=torch.load(R/"phase5/ternary_best.pt",map_location="cpu",weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    meta=read(R/"phase6/model_metadata.json")
    sizes=[0,0]
    for name,shape in SHAPES.items():
        layer=getattr(model,name)
        w=layer.parametrizations.weight.original.detach().numpy()
        info=meta["layers"][name]
        check(w.shape==shape and tuple(info["shape"])==shape,f"{name} layout")
        v=w.astype(np.float64)
        threshold=.7*np.abs(v).mean()
        q=np.where(v>threshold,1,np.where(v< -threshold,-1,0)).ravel()
        active=np.abs(v.ravel()[q!=0])
        scale=float(active.mean()) if len(active) else 0.
        np.testing.assert_allclose([info["alpha"],info["threshold"]],[scale,threshold],rtol=2e-6,atol=1e-8)
        np.testing.assert_array_equal(info["bias"],layer.bias.detach().numpy())
        for j,(fmt,base,width) in enumerate([("two_bit",4,4),("five_trit",3,5)]):
            packed=(R/f"phase6/{name}_{fmt}.bin").read_bytes()
            # Generate canonical bytes independently, including zero high padding.
            raw=bytes(sum((int(code)+1)*base**k for k,code in enumerate(q[i:i+width]))
                      for i in range(0,len(q),width))
            check(raw==packed,f"{name} {fmt} byte mismatch")
            sizes[j]+=len(raw)
    check(sizes==[182,147],"Unexpected weight size")
    w=torch.tensor([-2.,-1.,0.,1.,2.],requires_grad=True)
    TernaryWeight()(w).backward(torch.tensor([2.,3.,4.,5.,6.]))
    np.testing.assert_array_equal(w.grad.numpy(),[0,3,4,5,0])
    return "Verified 728 codes, both canonical byte streams, three scales, thresholds, 26 biases and bounded STE gradients."


def arithmetic():
    header=(R/"phase7/integer_config.h").read_text()
    def const(name):
        return int(re.search(r"#define\s+"+name+r"\s+\(?(-?\d+)",header).group(1))
    limit,unit,center,gain=[const(n) for n in ("ECG_LIMIT","ECG_ONE20","ECG_MEAN_UV","ECG_INPUT_GAIN")]
    check(limit==2**20 and unit==2**20 and gain>0,"Unexpected fixed-point configuration")
    max_input=max(abs(-2**31-center),abs(2**31-1-center))
    check(max_input*gain+unit//2 < 2**63,"Input normalization overflow")
    check(45*limit+22 < 2**31,"Global pooling overflow")
    for name,shape in SHAPES.items():
        scale=int(re.search(name+r"_scale\s*=\s*(\d+)",header).group(1))
        biases=list(map(int,re.search(name+r"_bias_q\[\]\s*=\s*\{([^}]+)",header).group(1).split(",")))
        dot=math.prod(shape[1:])*limit
        check(dot<2**31 and dot*scale+unit//2<2**63 and
              (dot*scale+unit//2)//unit+max(map(abs,biases))<2**63,f"{name} accumulator overflow")
    return "Integer bounds checked for every int32 input, normalization, all three dot products, bias, rounding and global averaging."


def benchmark():
    m=read(R/"phase2_dataset.json")
    labels=[]
    count=0
    with (R/"phase10/inputs.bin").open("rb") as inputs:
        for rid,row in sorted(m["records"].items()):
            if row["split"]!="test": continue
            with np.load(row["file"]) as d:
                y=d["y"]
                raw=np.rint(d["x"].astype(np.float64)*1000).astype(np.int32).tobytes()
                check(inputs.read(len(raw))==raw, f"Benchmark input/order mismatch at {rid}")
                labels.extend(y.tolist())
                count+=len(y)
        check(not inputs.read(1),"Unexpected trailing benchmark data")
    y=np.array(labels)
    r=read(R/"phase10/report.json")
    rows={v["name"]:v for v in r["variants"]}
    check(set(rows)==set(NAMES) and len(r["variants"])==6 and count==r["samples"]==18823,"Benchmark coverage")
    reference=None
    for name,expected_bytes in zip(NAMES,[3016,856,844,298,263,263]):
        row=rows[name]
        a=np.loadtxt(R/f"phase10/{name}_raw.txt",dtype=np.int64,ndmin=2)
        check(a.shape==(count,4) and np.all(a[:,0]>0) and np.isin(a[:,3],[0,1]).all(),f"{name} malformed rows")
        us=a[:,0]/1000
        matrix=np.zeros((2,2),dtype=int)
        np.add.at(matrix,(y,a[:,3]),1)
        np.testing.assert_array_equal(matrix,row["confusion_matrix"])
        diag=matrix.diagonal()
        checks=dict(accuracy=diag.sum()/count,macro_f1=np.mean(2*diag/(matrix.sum(0)+matrix.sum(1))),
                    V_recall=matrix[1,1]/matrix[1].sum(),mean_us=us.mean(),median_us=np.median(us),
                    p95_us=np.quantile(us,.95),serial_inferences_s=1e6/us.mean(),parameter_bytes=expected_bytes)
        for key,value in checks.items():
            check(math.isclose(row[key],value,rel_tol=1e-10,abs_tol=1e-10),f"{name} {key} mismatch")
        if name=="unpacked":
            reference=a[:,1:].copy()
            np.testing.assert_array_equal(matrix,read(R/"phase8/report.json")["integer_test"]["confusion_matrix"])
        if name in ("pack2","pack5","cached5"):
            np.testing.assert_array_equal(a[:,1:],reference,err_msg=name)
    return "112,938 timed rows reconciled with labels, exact input bytes, confusion matrices, latency statistics and logical parameter sizes."


def firmware():
    p9=read(R/"phase9/report.json")
    for p in ("2","5"):
        root=R/f"phase9/pack{p}"
        info=p9["builds"][p]
        check((root/"ecg.bin").stat().st_size==info["image_bytes"],f"Pack {p} firmware size")
        sizes=subprocess.check_output(["arm-none-eabi-size",str(root/"ecg.elf")],text=True)
        _,data,bss=map(int,sizes.splitlines()[-1].split()[:3])
        check(data+bss==info["static_ram_bytes"],"ELF static RAM mismatch")
        run=subprocess.run(["qemu-system-arm","-M",p9["machine"],"-nographic","-monitor","none","-serial","none",
                            "-semihosting-config","enable=on,target=native","-kernel",str(root/"ecg.elf")],
                           capture_output=True,text=True,timeout=60)
        check(run.returncode==0 and "ECG_ARM_PASS" in (run.stdout+run.stderr).splitlines(),f"Pack {p} emulator failure")
    return "Re-ran both ARM ELF self-tests on QEMU; reconciled image bytes and linked static RAM."


TESTS=[
    ("Phase and checkpoint identity",evidence,"critical","Check the first missing or inconsistent upstream artifact and regenerate its dependent phases."),
    ("Patient separation and inputs",split,"critical","Restore the Phase 2 manifest and shards, then regenerate affected training and evaluations."),
    ("Quantizer, STE and exporter",quantization,"critical","Inspect the named layer or gradient before regenerating packed weights."),
    ("Integer arithmetic bounds",arithmetic,"critical","Correct the fixed-point scale or widen the affected operation, then repeat numerical verification."),
    ("Raw benchmark reconciliation",benchmark,"major","Inspect the named variant, input ordering and raw rows before repeating Phase 10."),
    ("Firmware execution and size",firmware,"major","Check the ARM toolchain, firmware artifacts and QEMU target; rerun Phase 9 if needed.")]
LIMITS=[
    ("Major","Hardware latency and energy","Only desktop timers and QEMU execution are available.","Report OptiPlex timing and emulator correctness. Physical-board latency and energy require measurements on a board."),
    ("Minor","Precision versus runtime effects","FP32 uses PyTorch and the other implementations use C. INT8 uses floating-point rescaling.","Describe these as end-to-end implementation comparisons. Use the four C ternary rows to compare representations."),
    ("Minor","Memory accounting","Linked data/BSS omits the stack. Linux RSS includes process overhead. The cache occupies 728 additional bytes.","Report parameter bytes, firmware bytes, working arrays, static RAM and RSS separately. Peak board stack remains unmeasured."),
    ("Minor","Timing variation","The benchmark uses one run in a fixed implementation order and warm cache.","Report per-beat median and P95 for this run. Run-to-run variability is unknown."),
    ("Informational","Scope of accuracy","Two annotation-selected beat classes, one database, nine test patients and one training seed.","Describe the N/V beat classification task and patient-wise split when citing accuracy.")]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,fn,severity,action in TESTS:
        try:
            detail=fn()
            row=dict(check=name,status="PASS",verification=detail)
        except Exception as exc:
            row=dict(check=name,status="FAIL",severity=severity,observed_issue=str(exc) or type(exc).__name__,
                     root_cause="Requires diagnosis from the failed check.",corrective_action=action)
        rows.append(row)
        print(f"{row['status']} | {name}",flush=True)
    status="PASS" if all(r["status"]=="PASS" for r in rows) else "FAIL"
    payload=dict(status=status,checks=rows,limitations=[dict(severity=s,issue=i,cause=c,reporting_action=a) for s,i,c,a in LIMITS])
    md=["# Phase 11 engineering audit","",f"Overall verification **{status}**","","## Verification",""]
    for r in rows:
        md.extend([f"### {r['check']} — {r['status']}","",r.get("verification") or
                   f"{r['severity'].title()} finding. {r['observed_issue']}\n\n{r['root_cause']} {r['corrective_action']}",""])
    md.extend(["## Scope and limitations",""])
    md.extend(f"- **{s} — {i}.** {c} {a}" for s,i,c,a in LIMITS)
    for name,body in [("audit.md","\n".join(md)+"\n"),("report.json",json.dumps(payload,indent=2)+"\n")]:
        path=OUT/name
        tmp=path.with_suffix(".tmp")
        tmp.write_text(body)
        tmp.replace(path)
    print(f"Phase 11 {status} | results/phase11/report.json")
    return int(status!="PASS")


if __name__=="__main__":
    sys.exit(main())
