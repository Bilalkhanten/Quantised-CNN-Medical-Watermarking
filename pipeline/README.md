# Reproducibility package — IoMT medical-image watermarking (encoder / decoder / tamper localiser)

Code that trains the proposed system, evaluates it under the protocol of Section 3.9 and writes the **measured** values into Tables 10–14 of the manuscript.
It contains **no results**: every number must come from running it on the real data.

## What was verified where it was written (CPU, synthetic data only)
* Architecture reproduces Table 6 exactly: encoder 529,425 params / 77.59 MMACs; decoder 587,589 / 105.60; localiser 51,489 / 299.89; pruning at 0.38 gives 15 body channels (526,959 params).
* Payload chain (metadata → SHA-256 → AES-128-CBC or HMAC → 385 bits → BCH(511,385,t=14) → permutation; R_K from AES-CTR) round-trips; wrong key / wrong metadata / corrupted codeword give *unreadable* or *metadata mismatch*.
* All four training stages, pruning, fake-quant QAT, evaluation, baselines, statistics, DeLong test, cost script, ONNX export and docx filling run end to end on a 1-epoch synthetic run.
* ONNX FP32 graphs match PyTorch to ~1e-7 (encoder) / ~4e-7 (decoder).
* **Not verified: that training converges, and every performance number.**  In 300 CPU steps the message loss stayed at the chance plateau (0.693) in all configurations tried (full model, decoder alone, easy hand-made codes) — typical for 511-bit watermark training, but unproven here.
  Run `python train.py ... --probe 50` for the first epoch on the GPU: `L_msg` must fall clearly below 0.693 within a few hundred steps.  If it does not, try `--msg_warmup 3` (image-loss weight ramps 0→1; **not** paper-faithful, so report it if used), a lower `--omega`, or more Stage-1 epochs, before spending days on the full protocol.

## Data layout
* MIAS: folder with `mdb001.pgm … mdb322.pgm` and `Info.txt` (abnormality circles).  Patient = (n+1)//2; split 70/15/15 **by patient** (113/24/24), `--split_seed`; `split.json` is written to every run.
* Extra sets (CBIS-DDSM, ChestX-ray14, BraTS, LIDC-IDRI): folders of png/jpg/tif/dcm.  Training sets (`--pretrain`) and the held-out cross-dataset test images (`--cross NAME=dir`) must be disjoint — split them yourself.
* Images are resized to 256 × 256, 8-bit grey.  Tamper masks are exact (generated from the tamper operation); MIAS abnormality circles define the ROI for the weighted loss (ω = 5).

## Pipeline
| step | command | output |
|---|---|---|
| train (4 stages) | `python train.py --mias DIR --out runs/full_s0 --seed 0` | `stage3.pt` (FP32), `final_int8sim.pt`, `train_log.jsonl`, `config.json`, `split.json` |
| evaluate | `python evaluate.py --run runs/full_s0 --mias DIR [--cross CBIS=… BraTS=…] [--which fp32] [--denoiser dncnn.jit]` | `eval.json` (per-image arrays: BER, exact, PSNR, SSIM, Dice/IoU/TDR/FPR, auth AUC, security tests) |
| baselines | `python baselines.py dwtsvd|hidden --mias DIR --out runs/base_X.json` | same format |
| statistics | `python stats.py summary runs/full_s*/eval.json`, `python stats.py compare A.json B.json` | mean ± SD over seeds, patient-bootstrap CI, paired Wilcoxon + Holm |
| diagnostic impact | `python diag_classifier.py --run runs/full_s0 --mias DIR` | ROC-AUC original vs watermarked, DeLong p |
| INT8 export | `python export_onnx.py --run runs/full_s0 --calib DIR --alpha <calibrated alpha>` | `onnx/enc_int8.onnx` … |
| device cost | on the Pi: `python measure_cost.py --run runs/full_s0 --onnx_dir runs/full_s0/onnx --n 200 --power_csv trace.csv --idle_w 2.7` | `cost.json` |
| manuscript | `python fill_docx.py --docx paper.docx --out paper_measured.docx --full … --fp32 … --baseline … --abl … --cost … [--strip_labels]` | Tables 10–14 filled |
Everything above is chained by `run_all.sh` (set `MIAS`, `DOCX`, optionally `PRETRAIN`, `CROSS`, `DENOISER`, `COST`).

## Operating point and protocol details
* Fidelity: α is calibrated by bisection on the *validation* split so that mean PSNR = 40 dB (after 8-bit rounding); it is recorded in `eval.json`.  All methods are compared at this point.
* Exact recovery = BCH decode + AES/HMAC verification + metadata match.  With `--no_bch` it is exact match of the 385 uncoded bits.
* Attacks: Table 7 for training; the Table 10 grid (10 settings) plus unseen speckle, sharpening, contrast, JPEG 2000 and a denoiser for testing.  JPEG uses real `cv2` JPEG at test time.  The denoising row uses non-local means unless a trained network is passed with `--denoiser` (TorchScript, 1×1×H×W in [0,1]); the table label is changed to DnCNN **only then**.
* Security tests (`eval.json → security`): wrong key, 1-bit key change, transplant (payload / localiser), replay (freshness window), removal, re-embedding without K, random-bit forgery.
* Seeds: 5 (init, batch order, payload keys, attack draws).  Patient-level bootstrap resamples patients, not images.

## What is deliberately not here
* **ReDMark and MBRS are not re-implemented.**  Use the authors' official code, wrap it with `embed(img, m, rk)` / `extract(att)` (and an `.enc.alpha` strength knob) and call `baselines.run_eval()`; save to `runs/base_redmark.json`, `runs/base_mbrs.json`.  Until then those rows read **n/m**.
* **DnCNN** weights are not included; train one (σ = 15) or load a published one and export it with `torch.jit.trace`.
* **Energy** needs a power meter: log `t_seconds,watts` to CSV on the device, run `measure_cost.py` with `--power_csv/--idle_w`.  Without it energy/power cells read **n/m**.  The BLE radio row and the peak-memory column of Table 13 are analytic (airtime × power; tensor sizes) and are left as in the manuscript.
* **INT8 speed is not guaranteed.**  PyTorch "int8sim" is fake-quantisation (FP32 compute).  Only the exported ONNX INT8 graph, run on the target CPU, gives INT8 latency.  On a plain x86 server the INT8 graph was *slower* than FP32 (23 ms vs 10 ms on the smoke model); on ARM (Pi 4/5) it may differ — measure, do not assume.  In Table 13 the encoder rows are the ONNX graph minus the separately timed Haar DWT/IDWT (which the graph contains).
* Text in the manuscript that quotes numbers (abstract, discussion, conclusion, “Remaining illustrative values …”) is **not** rewritten automatically; `fill_docx.py` lists every paragraph that still mentions illustrative values so you can update them by hand from `runs/summary.txt`.

## Integrity note
`fill_docx.py` never leaves an old illustrative number beside measured ones: unfilled cells become `n/m`.  `--strip_labels` removes the “illustrative” caption text of a table only when every cell of that table was filled from a measurement.
