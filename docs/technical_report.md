# Ternary ECG inference on a desktop and an emulated STM32

**Muhammad Afnan · Software-only implementation · Audited OptiPlex run**

## Research question and result

Can a small ECG classifier store its learned weights as three symbols and still execute correctly on a microcontroller instruction set? The trained network stored 728 ternary weights in 147 bytes using five-trit packing. Its integer C implementation classified 18,823 held-out beats with 92.05% accuracy. The FP32 baseline achieved 96.95%. A post-training INT8 baseline achieved 95.86% and the lowest measured desktop median latency, 39.18 microseconds. Two packed ARM builds reproduced desktop integer results on five emulator vectors each.

The experiment adapts ternary-weight ideas to a two-class 1D CNN. The task starts at a database-supplied beat location. Beat detection, clinical validation, physical deployment, energy consumption and on-board timing are outside the measured scope.

## Data and experimental design

MIT-BIH Arrhythmia Database v1.0.0 provides the ECG waveforms and expert annotations. We used the MLII lead at 360 Hz and kept normal beats (N) and premature ventricular contractions (V). Each input contains 180 samples in millivolts, starting 90 samples before the annotation. Incomplete windows and other annotation types were excluded. The project used 44 records from 43 patients. The fixed patient-wise split kept records 201 and 202 together.

| Split | Patients | N | V | Beats |
|---|---:|---:|---:|---:|
| Training | 27 | 44,636 | 3,175 | 47,811 |
| Validation | 7 | 13,433 | 1,362 | 14,795 |
| Test | 9 | 16,458 | 2,365 | 18,823 |

Training windows determined the normalization mean and population standard deviation. Both models used the same initial underlying weights, training order, seed 42, batch size 128, Adam learning rate 0.001 and 20 epochs. Inverse-frequency loss weights addressed training imbalance. The earliest epoch attaining the highest validation macro-F1 supplied each model's final checkpoint. Precision, recall, F1, confusion matrices and epoch histories are in results/phase5/report.json.

## Network and numerical representations

| Operation | Output per beat | Parameters |
|---|---|---:|
| MLII input | 1 × 180 | 0 |
| Conv1, kernel 7, stride 2, padding 3, then ReLU | 8 × 90 | 64 |
| Max pool, width and stride 2 | 8 × 45 | 0 |
| Conv2, kernel 5, stride 1, padding 2, then ReLU | 16 × 45 | 656 |
| Global temporal mean and flatten | 16 | 0 |
| Linear classifier | 2 logits | 34 |

For each weight tensor, the forward pass sets the threshold to 0.7 times the mean absolute latent weight. Values strictly above or below the positive or negative threshold receive +1 or −1. Ties and values between the thresholds receive zero. The scale is the mean magnitude of the selected nonzero weights, or zero when none are selected. It minimizes squared reconstruction error for the chosen codes. Multiplying each code by its layer scale produces the effective weight. The surrogate backward pass propagates the incoming gradient when the latent weight has magnitude at most one. Outside this interval its gradient is zero.

All 754 parameters use FP32 in the baseline, for 3,016 bytes. The ternary model has 728 weight codes, 26 biases and three scales. Storing each code as its value plus one permits four 2-bit digits or five base-three digits per byte, least significant digit first. Each layer starts on a byte boundary. The two formats use 182 and 147 weight bytes respectively. Including biases and scales gives 298 and 263 bytes. At this network size the five-trit payload is 11.47 times smaller than the FP32 payload. “1.58 bits” refers to log2(3) per ternary symbol; the actual five-trit weight payload here uses 1.615 bits per weight including padding.

## Integer inference and verification

The standalone C engine reads 180 integer microvolt samples. Normalization, signed weight accumulation, bias, ReLU, pooling and the two output scores use integer arithmetic. Activations and scores have 12 fractional bits, scale multipliers use 20, and widened intermediates protect scaling operations. Bounds were checked for all signed 32-bit input values. Class 0 wins a score tie. The engine uses fixed-size working arrays.

Both packing formats reproduced all trained weight codes. On 46 selected test beats, both C builds matched the Python integer reference exactly at six stages. Across the full test set the two packing formats agreed on both scores for every beat. Integer C and the PyTorch ternary checkpoint differed on five decisions. Their largest absolute logit difference was 0.00502515.

| Implementation | True N predicted N | True N predicted V | True V predicted N | True V predicted V | Macro-F1 | V recall |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 15,960 | 498 | 76 | 2,289 | 0.9355 | 0.9679 |
| PyTorch ternary | 15,061 | 1,397 | 100 | 2,265 | 0.8521 | 0.9577 |
| Integer C ternary | 15,062 | 1,396 | 100 | 2,265 | 0.8522 | 0.9577 |

Relative to FP32, integer ternary inference missed 24 additional V beats and classified 898 additional normal beats as V. Most of the classification difference appeared during ternary training and weight quantization. Converting the trained ternary network to integer C changed five decisions and improved the total correct count by one.

## Desktop benchmark and emulated firmware

The benchmark pinned execution to one CPU core, used batch size one and timed normalization through the class decision after 32 warm-up calls. Each implementation processed the same 18,823 test beats in record order. The INT8 variant used signed 8-bit weights and activations with integer dot products and floating-point rescaling. Its activation scales came from training data.

| Implementation | Accuracy | Median (µs) | P95 (µs) |
|---|---:|---:|---:|
| FP32 PyTorch | 96.95% | 312.93 | 384.92 |
| Conventional INT8 C | 95.86% | 39.18 | 44.28 |
| Unpacked ternary C | 92.05% | 64.78 | 70.87 |
| 2-bit packed C | 92.05% | 93.71 | 100.58 |
| Five-trit packed C | 92.05% | 237.94 | 253.65 |
| Cached five-trit C | 92.05% | 67.02 | 73.02 |

Caching required 728 extra RAM bytes and reduced five-trit median latency by 71.8%. Among the ternary implementations, unpacked weights gave the lowest median latency. The FP32 timing includes PyTorch dispatch, so the ratio between FP32 and C combines runtime and representation effects. These are single-run desktop measurements. Raw timing samples, mean latency, throughput and process RSS are preserved in results/phase10/.

Both STM32F405 Cortex-M4 firmware builds ran in QEMU on the netduinoplus2 target. Each reproduced both scores and the class for two real validation beats and three synthetic vectors. The complete 2-bit image was 5,848 bytes and the five-trit image was 5,884 bytes, including the test harness and embedded inputs. The smaller five-trit weight array therefore did not yield the smaller complete image. Both ELF files had zero linked data/BSS bytes. The linker reserved 8 KiB for stack availability; peak stack use was unmeasured.

## Audit and limits

The Phase 11 audit passed all six groups. It reconciled checkpoint identities, patient separation, weights and surrogate gradients, arithmetic bounds, all 112,938 timed rows, and both firmware self-tests. The release includes the detailed audit findings. The experiment covered one dataset, two annotation-defined classes, one training seed and nine test patients. Physical-board latency, power, peak stack use and variation between benchmark runs remain open measurements.

## Sources

- Moody and Mark. MIT-BIH Arrhythmia Database v1.0.0. PhysioNet. Data licensed under Open Data Commons Attribution v1.0. https://doi.org/10.13026/C2F305
- Li et al. Ternary Weight Networks. https://arxiv.org/abs/1605.04711
- Ma et al. The Era of 1-bit LLMs. https://arxiv.org/abs/2402.17764
- Wang et al. Bitnet.cpp. https://arxiv.org/abs/2502.11880
- QEMU. STMicroelectronics STM32 boards. https://www.qemu.org/docs/master/system/arm/stm32.html
