# Implementation package – secure, energy-efficient medical-image watermarking for IoMT

Quantised CNN encoder (Haar-LL embedding), spatial-transformer blind decoder and U-Net tamper localiser, with the cryptographic payload chain
(SHA-256 → AES-128-CBC or HMAC-SHA-256 → BCH(511,385,t=14) → keyed permutation), the differentiable attack layer, four-stage training, network slimming,
INT8 quantisation-aware training, evaluation protocol, baselines, statistics, ONNX export, device cost measurement and the script that writes measured
results into the manuscript tables.

## Read this first
* **This package contains code and documents, not results.** No model has been trained on real data. Every learned-model number in the manuscript
  (Tables 10–14, Table 15 and the ablation figure) is an *illustrative demonstration value*, not a measurement.
* Verified on CPU with synthetic data only: architecture sizes (Table 6), payload round trip, end-to-end execution of the training stages,
  evaluation, ONNX export and docx filling. **Not verified: that training converges.** A short CPU test stayed at the chance plateau; run the
  `--probe` step on a GPU first (see `pipeline/README.md`).
* Real data (MIAS, CBIS-DDSM, ChestX-ray14, BraTS, LIDC-IDRI) and the Raspberry Pi power measurements are not included and must be supplied.
* ReDMark and MBRS baselines are **not** implemented (rows stay "n/m" until you add them). See `pipeline/README.md`.

## Layout
| folder | content |
|---|---|
| `pipeline/` | the whole experiment: `models.py`, `system.py`, `payload.py`, `attacks.py`, `data.py`, `train.py`, `evaluate.py`, `baselines.py`, `stats.py`, `diag_classifier.py`, `export_onnx.py`, `measure_cost.py`, `fill_docx.py`, `run_all.sh`, figure scripts and the README with the full protocol |
| `figures/` | `fig_ablation_components.py` – ablation figure from the Table 12 values |
| `docs_tools/` | generators for the Table 15 comparison document and the point-by-point response letter |
| `deliverables/` | revised manuscript (with Table 15), response letter, Table 15 document, figures, Tables 1–4 citation audit |
| `smoke_test.sh` | 1-minute integrity check (no data needed) |

## Quick start
```bash
pip install -r requirements.txt
./smoke_test.sh                      # package integrity
cd pipeline
MIAS=/path/to/mias DOCX=/path/to/manuscript.docx ./run_all.sh   # full chain: train → evaluate → baselines → stats → export → fill tables
```
Seeds 0–4, patient-level 70/15/15 split, α calibrated to 40 dB on the validation split, patient-level bootstrap CIs, Holm-corrected paired Wilcoxon,
DeLong test for the authentication AUC (details in `pipeline/README.md`).

## After you have real results
1. `run_all.sh` fills Tables 10–14 from the runs; cells without data become `n/m`.
2. Re-run `figures/fig_ablation_components.py` after editing the values in it, and remove the "illustrative" banner.
3. Update the abstract, Discussion, Conclusion and Table 15 by hand from `runs/summary.txt`; remove the † marks.
4. Change the statuses in `docs_tools/build_response_letter.py` from "Open" to "Addressed" only for items that now have measured evidence.
