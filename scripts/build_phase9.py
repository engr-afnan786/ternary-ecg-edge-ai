"""Build and execute STM32F405 Cortex-M4 firmware on QEMU, from the project root."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

OUT = Path("results/phase9")
PLAT = Path("embedded/platform/stm32f405")
BOARD = "netduinoplus2"


def read(path):
    return json.loads(Path(path).read_text())


def run(args, **kwargs):
    return subprocess.check_output(list(map(str,args)), text=True, **kwargs)


def write_vectors():
    p7 = read("results/phase7/report.json")
    p8 = read("results/phase8/report.json")
    meta = read("results/phase6/model_metadata.json")
    digest = hashlib.sha256(Path(meta["source_checkpoint"]).read_bytes()).hexdigest()
    if not (p7["status"] == p8["status"] == "PASS" and
            digest == meta["sha256"] == p7["checkpoint_sha256"] == p8["checkpoint_sha256"]):
        raise ValueError("Phase reports or checkpoint disagree")
    beats = []
    for src in meta["golden_validation_sources"]:
        rid = src["record"]
        with np.load(f"datasets/processed/mitdb_nv_v1/validation/{rid}.npz") as d:
            pos = np.flatnonzero((d["sample"]==src["sample"]) & (d["y"]==src["label"]))
            if len(pos) != 1:
                raise ValueError(f"Validation source mismatch in {rid}")
            beats.append(np.rint(d["x"][pos[0],0].astype(np.float64)*1000).astype(np.int64))
    beats.extend([np.zeros(180,dtype=np.int64),np.arange(-90,90)*25,np.tile([-2**31,2**31-1],90)])
    expected = p7["expected_scores_and_class"]
    if np.shape(beats) != (5,180) or np.shape(expected) != (5,3):
        raise ValueError("Unexpected Phase 7 vector layout")
    def rows(values):
        return ",\n".join("{"+", ".join(map(str,row))+"}" for row in values)
    (OUT/"phase9_vectors.h").write_text("#define VECTOR_COUNT 5\nstatic const int32_t inputs[5][180]={\n"+
                                        rows(beats)+"};\nstatic const int32_t expected[5][3]={\n"+rows(expected)+"};\n")
    return digest


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    dest = OUT/"report.json"
    dest.unlink(missing_ok=True)
    gcc, objcopy, size, qemu = "arm-none-eabi-gcc","arm-none-eabi-objcopy","arm-none-eabi-size","qemu-system-arm"
    for tool in (gcc,objcopy,size,qemu):
        if not shutil.which(tool):
            raise SystemExit(f"Missing {tool}. Install the Phase 9 Ubuntu packages.")
    if BOARD not in run([qemu,"-machine","help"]):
        raise SystemExit(f"Installed QEMU lacks {BOARD}")
    digest = write_vectors()
    builds = {}
    for packing in (2,5):
        root = OUT/f"pack{packing}"
        root.mkdir(exist_ok=True)
        elf = root/"ecg.elf"
        image = root/"ecg.bin"
        run([gcc,"-mcpu=cortex-m4","-mthumb","-mfloat-abi=soft","-std=c11","-Os","-ffreestanding",
             "-ffunction-sections","-fdata-sections","-fno-unwind-tables","-fno-asynchronous-unwind-tables",
             "-nostdlib",f"-DECG_PACKING={packing}","-Iembedded/include","-Iresults/phase6","-Iresults/phase7",
             "-I"+str(OUT),PLAT/"startup.S",PLAT/"firmware.c","embedded/src/ecg_integer.c",
             "-T",PLAT/"memory.ld","-Wl,--gc-sections","-Wl,--build-id=none",
             "-Wl,-Map="+str(root/"ecg.map"),"-lgcc","-o",elf])
        run([objcopy,"-O","binary",elf,image])
        sizes = run([size,elf])
        (root/"size.txt").write_text(sizes)
        _,data,bss = map(int,sizes.splitlines()[-1].split()[:3])
        result = subprocess.run([qemu,"-M",BOARD,"-nographic","-monitor","none","-serial","none",
                                 "-semihosting-config","enable=on,target=native","-kernel",str(elf)],
                                capture_output=True,text=True,timeout=60)
        log = result.stdout + result.stderr
        (root/"qemu.log").write_text(log)
        if result.returncode != 0 or "ECG_ARM_PASS" not in log.splitlines():
            raise RuntimeError(f"QEMU verification failed for packing {packing}: {log}")
        builds[packing] = dict(image_bytes=image.stat().st_size,static_ram_bytes=data+bss,
                               stack_reserved_bytes=8192,checked_vectors=5,elf=str(elf))
        print(f"Packing {packing}: QEMU PASS | 5 vectors | image {image.stat().st_size} bytes | static RAM {data+bss} bytes")
    report = dict(status="PASS",checkpoint_sha256=digest,machine=BOARD,builds=builds,
                  compiler=run([gcc,"--version"]).splitlines()[0],qemu=run([qemu,"--version"]).splitlines()[0])
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2))
    tmp.replace(dest)
    print("Phase 9 PASS |",dest)


if __name__ == "__main__":
    main()
