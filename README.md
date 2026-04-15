# Paradigm-Agnostic cWGAN-GP for EEG Data Augmentation

A unified conditional Wasserstein GAN with gradient penalty (cWGAN-GP) that generates
physiologically plausible synthetic EEG across three distinct BCI paradigms —
**Motor Imagery (MI)**, **P300**, and **Resting-State (RS)** — using a single
minimal generator/critic architecture paired with a spectral consistency loss.

The goal is to challenge the current trend of paradigm-specific GAN designs: one
architecture, three paradigms, four public datasets, with augmentation-driven gains
measured on downstream classification accuracy.

---

## Highlights

- **Single architecture**, three paradigms — no per-paradigm tweaks beyond input shape.
- **Wasserstein + spectral loss only.** Covariance, autocorrelation, and alpha-band
  auxiliary losses were ablated and removed — they hurt quality on small EEG sets.
- **Paradigm-aware normalization.** MI/RS use global min-max; P300 uses per-channel
  z-score with ±3σ clipping. Normalization stats are persisted and inverted after
  generation so synthetic signals match real µV scales.
- **RMS gain correction** post-generation to compensate tanh compression of the
  generator output.
- Evaluation tied to each paradigm: CSP-SVM accuracy + ERD/ERS maps for MI,
  LDA accuracy + ERP waveform comparison for P300, PSD + alpha-band statistics
  across Young vs Elderly groups for RS.

## Datasets

| Paradigm | Dataset             | Subjects | Channels | Fs (Hz) | Task |
|----------|---------------------|----------|----------|---------|------|
| MI       | BCI Competition IV 2a | 9      | 12 / 22  | 250     | Left vs right hand |
| MI       | Cho2017             | 52       | 12 / 64  | 250     | Left vs right hand |
| P300     | P300 BCI Speller    | 12 × 4 pairs | 8    | 250     | Target vs non-target |
| RS       | Young / Elderly     | 22       | 8        | 125     | Eyes open / closed, pre/post |

Raw datasets are **not tracked** in this repository. Contact the author for access
details or follow the download instructions in each paradigm's original reference.

## Repository layout

```
.
├── README.md                       # You are here
├── EEG_scripts/
│   ├── pyproject.toml              # Poetry environment: eeg-cwgan
│   ├── poetry.lock
│   ├── scripts/                    # Training, evaluation, figure generation
│   │   ├── nbase.py                # Canonical cWGAN-GP trainer (all paradigms)
│   │   ├── MI_pipeline.py          # MI training + evaluation loop
│   │   ├── P300_pipeline.py        # P300 training + evaluation loop
│   │   ├── P300_pipeline_upsample.py
│   │   ├── P300_training.py
│   │   ├── RS_analysis.py          # RS PSD + group-level analysis
│   │   ├── generate_MI_figures.py  # Publication figures — MI
│   │   ├── generate_P300_figures.py# Publication figures — P300
│   │   ├── evaluate_classification.py
│   │   ├── GNN.py                  # Graph-based comparison baseline
│   │   ├── tl_gan_m.py             # Transfer-learning variant
│   │   └── plots.py
│   └── figures/                    # Generated publication figures
│       ├── mi_bci2a/
│       ├── mi_cho2017/
│       ├── p300/visualization_P300/
│       └── rs/
```

## Getting started

```bash
# 1. Clone
git clone <repo-url> && cd Python_Codes/EEG_scripts

# 2. Install the environment (Poetry)
poetry install

# 3. Point the scripts at your local data directory
#    (see the path constants near the top of each pipeline script)

# 4. Train a paradigm
poetry run python scripts/MI_pipeline.py
poetry run python scripts/P300_pipeline.py
poetry run python scripts/RS_analysis.py

# 5. Regenerate figures
poetry run python scripts/generate_MI_figures.py
poetry run python scripts/generate_P300_figures.py
```

### Technical stack

Python 3.10 · TensorFlow/Keras (CUDA) · MNE-Python · scikit-learn · pyRiemann ·
matplotlib / seaborn. GPU training validated on an NVIDIA RTX 4060 (8 GB).

## Results at a glance

Generated figures under `EEG_scripts/figures/` cover:

- **MI** — CSP pattern topoplots, ERD/ERS time-frequency maps, grand-average PSD,
  per-subject accuracy box plots (real vs augmented).
- **P300** — Cz ERP with ± SEM ribbons, peak latency and amplitude distributions,
  LDA accuracy scatter across 48 subject/pair blocks.
- **RS** — Grand-average PSD with alpha-band (8–13 Hz) highlights, Young vs Elderly
  comparisons pre/post session.

Training artefacts (checkpoints, `.mat` outputs, subject-level CSVs) and raw
recordings are intentionally excluded from version control.

## Author

**Sourojit Goswami** — PhD Student, School of Electrical and Electronic Engineering,
University of Sheffield. Supervisors: Prof. Sean Anderson, Dr. Mahnaz Arvaneh.

Work connected to a scoping review submitted to *IEEE Transactions on Biomedical
Engineering* on generative models for EEG, and a forthcoming journal manuscript on
the paradigm-agnostic approach introduced here.

## License

Released for academic use. Please cite the accompanying publications if you build
on this work. Contact the author before redistributing or adapting for commercial
purposes.
