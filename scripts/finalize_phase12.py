"""Package audited ECG sources and evidence. Run from the existing project root."""
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

R=Path("results")
OUT=R/"phase12"
RELEASE=OUT/"ternary_ecg_release.zip"


def read(path):
    return json.loads(Path(path).read_text())


def sha(stream):
    h=hashlib.sha256()
    for data in iter(lambda:stream.read(1024*1024),b""):
        h.update(data)
    return h.hexdigest()


def hashed(path):
    with path.open("rb") as stream:
        return sha(stream)


def write_archive(target, files, extras):
    """Hash original files, then independently hash the decompressed archive."""
    index={}
    with ZipFile(target,"w",ZIP_DEFLATED,compresslevel=3) as archive:
        for name in sorted(files):
            path=Path(name)
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Required regular file missing: {name}")
            index[name]=hashed(path)
            archive.write(path,name)
        for name,body in extras.items():
            if name in index:
                raise ValueError(f"Duplicate archive path: {name}")
            data=body.encode()
            index[name]=hashlib.sha256(data).hexdigest()
            archive.writestr(name,data)
        manifest="".join(f"{value}  {name}\n" for name,value in sorted(index.items()))
        archive.writestr("SHA256SUMS.txt",manifest)
    with ZipFile(target) as archive:
        if archive.read("SHA256SUMS.txt").decode()!=manifest or set(archive.namelist())!=set(index)|{"SHA256SUMS.txt"}:
            raise ValueError("Archive inventory mismatch")
        for name,expected in index.items():
            with archive.open(name) as stream:
                if sha(stream)!=expected:
                    raise ValueError(f"Archive verification failed: {name}")
    return len(index)


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    audit=read(R/"phase11/report.json")
    if audit["status"]!="PASS" or len(audit["checks"])!=6 or any(c["status"]!="PASS" for c in audit["checks"]):
        raise ValueError("Phase 11 audit is incomplete. See results/phase11/audit.md.")
    bench=read(R/"phase10/report.json")
    meta=read(R/"phase6/model_metadata.json")
    ck=hashed(R/"phase5/ternary_best.pt")
    if not ck==meta["sha256"]==bench["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from audited export")
    files=set("""phase1_check.py phase2_prepare.py docs/technical_report.html docs/technical_report.md docs/reproduce.md
    results/phase1_environment.json results/phase2_dataset.json results/phase3_fp32.json results/phase4_ternary.json
    results/phase5/fp32_best.pt results/phase5/ternary_best.pt results/phase5/report.json results/phase5/learning_curves.png
    results/phase6/model_config.h results/phase6/model_metadata.json results/phase6/model_statistics.json results/phase6/ternary_weights.h results/phase6/golden_test_vectors.h
    results/phase7/report.json results/phase7/example_beat_uv.txt results/phase7/integer_config.h results/phase7/ecg_pack2 results/phase7/ecg_pack5
    results/phase8/report.json results/phase9/report.json results/phase9/phase9_vectors.h results/phase10/report.json results/phase10/inputs.bin results/phase10/comparison.csv
    results/phase10/baseline8.h results/phase10/byte_weights.h results/phase11/report.json results/phase11/audit.md""".split())
    # Only collect project artifacts used by the documented pipeline.
    modules="check_fp32 check_ternary train_phase5 phase5_metrics phase6_codec export_phase6 build_phase7 check_phase8 build_phase9 bench_phase10 audit_phase11 finalize_phase12".split()
    files.update("scripts/"+m+".py" for m in modules)
    files.update("""src/models/ecg_fp32.py src/models/ecg_ternary.py src/quantization/ternary.py
    benchmarks/int8.c benchmarks/runner.c benchmarks/ternary.c embedded/include/ecg_integer.h embedded/src/ecg_integer.c embedded/src/ecg_cli.c
    embedded/platform/stm32f405/firmware.c embedded/platform/stm32f405/startup.S embedded/platform/stm32f405/memory.ld""".split())
    for layer in ("conv1","conv2","head"):
        for fmt in ("two_bit","five_trit"):
            files.add(f"results/phase6/{layer}_{fmt}.bin")
    for p in (2,5):
        files.update(f"results/phase9/pack{p}/{name}" for name in ("ecg.elf","ecg.bin","ecg.map","size.txt","qemu.log"))
    for row in bench["variants"]:
        name=row["name"]
        files.add(f"results/phase10/{name}_raw.txt")
        if name!="fp32": files.add(f"results/phase10/{name}")
    for rid,row in read(R/"phase2_dataset.json")["records"].items():
        name=f"datasets/processed/mitdb_nv_v1/{row['split']}/{rid}.npz"
        if row["file"]!=name or not rid.isdigit() or row["split"] not in ("train","validation","test"):
            raise ValueError("Unexpected dataset path")
        files.add(name)
    results=["# Results from the packaged run","","| Variant | Accuracy | Macro-F1 | V recall | Mean µs | Median µs | P95 µs | Serial beats/s | Parameter bytes |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in bench["variants"]:
        results.append("| {name} | {accuracy:.4f} | {macro_f1:.4f} | {V_recall:.4f} | {mean_us:.2f} | {median_us:.2f} | {p95_us:.2f} | {serial_inferences_s:.1f} | {parameter_bytes} |".format(**row))
    p5=read(R/"phase5/report.json")
    for name,row in p5["models"].items():
        results.append(f"\n{name} selected epoch {row['best_epoch']}, validation macro-F1 {row['validation_macro_f1']:.4f}.")
    results.extend(["","## Trained ternary distribution","","| Layer | −1 | 0 | +1 | Zero % | Scale |","|---|---:|---:|---:|---:|---:|"])
    for name,r in p5["ternary_layers"].items():
        a,b,c=r["counts_minus_zero_plus"]
        results.append(f"| {name} | {a} | {b} | {c} | {r['zero_percent']:.2f} | {r['alpha']:.8g} |")
    versions="\n".join(sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata.get("Name")))
    req="--extra-index-url https://download.pytorch.org/whl/cpu\n"+"".join(f"{n}=={metadata.version(n)}\n" for n in ("torch","numpy","pandas","wfdb","matplotlib"))
    extras={"README.md":"# Ternary ECG classifier\n\nOpen [the technical report](docs/technical_report.html), [run results](docs/run_results.md), [reproduction guide](docs/reproduce.md) and [audit](results/phase11/audit.md).\n",
            "requirements.txt":req,"docs/run_results.md":"\n".join(results)+"\n",
            "docs/environment.txt":f"Python {sys.version}\n{platform.platform()}\n\n{versions}\n"}
    with TemporaryDirectory(dir=OUT) as temp:
        archive=Path(temp)/RELEASE.name
        n=write_archive(archive,files,extras)
        checksum=hashed(archive)
        result=dict(status="PASS",archive=str(RELEASE),verified_files=n,archive_bytes=archive.stat().st_size,
                    archive_sha256=checksum,ternary_checkpoint_sha256=ck)
        report=Path(temp)/"report.json"
        report.write_text(json.dumps(result,indent=2)+"\n")
        sums=Path(temp)/"release.sha256"
        sums.write_text(f"{checksum}  {RELEASE.name}\n")
        os.replace(archive,RELEASE)
        os.replace(sums,OUT/sums.name)
        os.replace(report,OUT/report.name)
    print(f"Verified {n} packaged files | archive SHA-256 {checksum}")
    print(f"Final project archive | {RELEASE}")
    print("Phase 12 PASS | results/phase12/report.json")


if __name__=="__main__":
    build()
