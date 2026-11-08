# Reproduce the ternary ECG experiment

For this source repository, start at **Set up a fresh environment**, then run the phase table from the repository root. The archive inspection and fresh-directory copy steps apply to a generated Phase 12 release.

Run commands from the root of the extracted project on Ubuntu x86-64. The release includes processed ECG windows derived from PhysioNet, trained weights, reports, source, desktop programs and ARM firmware. The original waveforms can be downloaded again using phase2_prepare.py. Retain the dataset attribution when sharing the archive. [Dataset and citation](https://physionet.org/content/mitdb/1.0.0/) · [Open Data Commons Attribution License](https://opendatacommons.org/licenses/by/1-0/)

## Inspect an existing release

Extract the release into a new directory. From that directory run:

```bash
sha256sum -c SHA256SUMS.txt --quiet
chmod +x results/phase7/ecg_pack2 results/phase7/ecg_pack5
./results/phase7/ecg_pack5 < results/phase7/example_beat_uv.txt
```

A successful checksum verification exits silently. The example output for the audited model is `18738 -17734 0`. Columns give the two Q12 logits and the predicted class. Divide each logit by 4096 for its real-valued equivalent. For other host architectures, rebuild with Phase 7 before executing.

## Set up a fresh environment

The audited setup used Ubuntu, Python 3.12 and CPU PyTorch. The release records the installed packages in docs/environment.txt. Its requirements.txt pins the direct dependencies from the packaging environment. Run on a compatible Python and Linux platform.

```bash
sudo apt update
sudo apt install -y --no-install-recommends build-essential python3-venv binutils gcc-arm-none-eabi binutils-arm-none-eabi qemu-system-arm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Replay the evidence or train from scratch

For an evidence check using included checkpoints, datasets and raw measurements, run `python -u -m scripts.audit_phase11`. For a fresh run, copy only source, requirements and documentation into a **new directory** using the commands below. This starts training at epoch one and preserves the original evidence, including its timing samples.

```bash
replay_dir=$(mktemp -d "$HOME/ternary-replay.XXXXXX")
cp -r phase1_check.py phase2_prepare.py scripts src benchmarks embedded docs requirements.txt "$replay_dir/"
cd "$replay_dir"
```

Keep the virtual environment activated. Run each command and verify its PASS result before proceeding.

| Phase | Command | Main evidence |
|---|---|---|
| 1. CPU and C environment | `python phase1_check.py` | Environment report |
| 2. Prepare dataset | `python -u phase2_prepare.py` | Patient split, counts, processed windows |
| 3. FP32 architecture | `python -m scripts.check_fp32` | Shapes, normalization and optimizer check |
| 4. Ternary architecture | `python -m scripts.check_ternary` | Ternary codes, scales and STE check |
| 5. Train and evaluate | `python -u -m scripts.train_phase5` | Best checkpoints, metrics, learning curves |
| 6. Export weights | `python -m scripts.export_phase6` | Headers, metadata and golden vectors |
| 7. Build desktop engine | `python -m scripts.build_phase7` | Two C executables and integer comparison |
| 8. Compare numerical results | `python -u -m scripts.check_phase8` | Test predictions and intermediate comparisons |
| 9. Build and emulate | `python -u -m scripts.build_phase9` | ARM images, maps, ELF sizes and QEMU logs |
| 10. Benchmark | `python -u -m scripts.bench_phase10` | CSV, raw samples and JSON |
| 11. Audit | `python -u -m scripts.audit_phase11` | Engineering audit |
| 12. Package | `python -u -m scripts.finalize_phase12` | Checksummed release ZIP |

The packaged technical report describes the original audited run. For a replay, use the new run's report files and docs/run_results.md in its archive for exact values. Timing varies with CPU and load. Retrying Phase 5 recovers the last completed epoch. If a PhysioNet download is interrupted, rerun Phase 2 to reuse verified raw files. If a checksum fails after archive extraction, extract a fresh copy before auditing it.
