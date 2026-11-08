# Ternary ECG inference for edge devices

A reproducible two-class ECG beat classifier built with PyTorch and standalone integer C. The project compares FP32, conventional INT8, and four representations of ternary weights. Two firmware builds ran on an STM32F405 emulated in QEMU.

## Measured results

The test set comprised 18,823 annotated normal and premature ventricular beats from nine held-out patients. Desktop timing used one CPU core and a batch size of one.

| Implementation | Accuracy | Median latency on the OptiPlex |
|---|---:|---:|
| FP32 PyTorch | 96.95% | 312.93 µs |
| Conventional INT8 C | 95.86% | 39.18 µs |
| Unpacked ternary C | 92.05% | 64.78 µs |
| Two-bit packed C | 92.05% | 93.71 µs |
| Five-trit packed C | 92.05% | 237.94 µs |
| Cached five-trit C | 92.05% | 67.02 µs |

Five-trit packing reduced parameter storage from 3,016 to 263 bytes including biases and scales. The integer C model detected 2,265 of 2,365 ventricular beats. Hardware energy and physical-board latency remain unmeasured.

## Start here

1. Install the Ubuntu tools and CPU Python environment using the [setup and reproduction guide](docs/reproduce.md).
2. Run `python phase1_check.py` and confirm both checks pass.
3. Run `python -u phase2_prepare.py` to download and prepare MIT-BIH records. Follow the remaining phase commands in order. Training produces the checkpoints and reports locally.

Read the [technical report](docs/technical_report.md) for the patient split, architecture, packing method, independent checks and limitations. View the [project poster](docs/project-poster.png).

## Data and attribution

Muhammad Afnan implemented this software-only ECG experiment. The dataset is the MIT-BIH Arrhythmia Database v1.0.0 by George Moody and Roger Mark, distributed through PhysioNet under the Open Data Commons Attribution License v1.0. The model classifies beats at supplied annotation locations. [Dataset and citation](https://doi.org/10.13026/C2F305) · [Dataset license](https://opendatacommons.org/licenses/by/1-0/)
