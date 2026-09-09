# Chest-CT-ZeroShot-Benchmark

Comparing a 3D-native medical vision-language model against a naive 2D slice-pooled
baseline on zero-shot multi-abnormality classification, using CT-RATE.

## Motivation

Most vision-language models operate on 2D images. Medical imaging is often
volumetric. This project asks a narrow, concrete question: **how much does
3D-native modeling actually buy you over pooling 2D slice embeddings**, on
a real clinical zero-shot classification task?

## Setup

- **Dataset**: [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)
  (chest CT volumes + 18 abnormality labels), 100-volume random subset of the
  validation split.
- **3D-native model**: [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP)
  (CTViT image encoder, BiomedVLP-CXR-BERT text encoder), pretrained checkpoint,
  zero-shot inference only — no fine-tuning.
- **2D baseline**: [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224),
  16 evenly-spaced axial slices per volume, mean-pooled to a volume-level embedding.
- **Prompting**: identical for both models — `"{finding} is present."` vs.
  `"{finding} is not present."`, softmax over the pair, per finding.
- **Preprocessing**: identical for both models up to the point of encoding —
  same HU clipping, same physical resampling, same crop/pad — so the only
  variable being tested is 3D-native vs. 2D-slice-pooled encoding.

## A preprocessing correction worth noting

CT-RATE has two data versions: the original volumes, and a corrected `_fixed`
version where intensity/spacing corrections are baked directly into the NIfTI
files. CT-CLIP's released preprocessing script assumes the *original*
uncorrected format and manually re-applies a rescale step. Running that
rescale step on `_fixed` volumes double-corrects the intensities. This repo's
`dataset_fixed.py` is a corrected fork of CT-CLIP's dataset class with that
step removed, verified against raw HU value ranges before running any
inference.

## Results

Zero-shot performance across 18 CT-RATE findings, 100-volume subset:

| Model | Mean AUC | Notes |
|---|---|---|
| CT-CLIP (3D) | ~0.70 | Comparable to the ~0.734 reported in the original paper on full validation |
| BiomedCLIP (2D, slice-pooled) | ~0.59 | |

Full per-finding breakdown in `results/ctclip_zeroshot/ctclip_metrics.csv` and
`results/biomedclip_zeroshot/biomedclip_metrics.csv`.

**A more interesting finding than the raw gap**: on some findings (e.g. pleural
effusion), BiomedCLIP's AUC is competitive with or even slightly exceeds
CT-CLIP's — but its F1 score collapses to near-zero. AUC measures ranking
ability; F1 depends on the 0.5 decision threshold being meaningful. This
points to a **calibration problem, not just a discrimination problem**:
BiomedCLIP retains real signal but is poorly calibrated for this specific
present/absent radiology framing, likely because it wasn't trained on this
prompt style or on volumetric CT data. CT-CLIP, trained natively on this
exact task formulation, is both a better ranker and better calibrated.

## Reproducing

1. Get CT-RATE access (gated on Hugging Face) and a read token.
2. `dataset_fixed.py` expects `data_volumes/`, `checkpoints/`, and the three
   CSVs (`validation_metadata.csv`, `validation_reports.csv`,
   `valid_labels.csv`) locally — see `EDA/get_csvs.py` and `EDA/get_subset.py`
   for the download scripts.
3. `src/chest_ct_zeroshot_benchmark/ctclip_zeroshot.py` and
   `biomedclip_zeroshot.py` run inference and save metrics independently.

## Limitations

- 100-volume subset, not the full validation set — some findings have few
  positive cases, so per-finding metrics for rare findings are noisy.
- Zero-shot only; no fine-tuning attempted on either model.
- BiomedCLIP baseline uses uniform mean-pooling across slices, not a learned
  aggregation — a more sophisticated pooling strategy might close some of the gap.