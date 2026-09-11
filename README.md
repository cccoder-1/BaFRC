# BaFRC

Official code layout for BaFRC experiments on FewRel and FS-TACRED.

## Installation

```bash
pip install -r requirements.txt
```

Download `bert-base-uncased` (or provide another compatible checkpoint) and
prepare the datasets under `data/fewrel` and `data/fs_tacred`.

The repository contains relation description pools but does not redistribute
the original datasets or trained checkpoints.

## Baseline experiments

Run commands from the repository root:

```bash
bash scripts/run_fewrel_baseline.sh
bash scripts/run_tacred_baseline.sh
```

Experiment settings can be overridden with the environment variables described
at the beginning of each script.

## Repository structure

- `models/`: BaFRC and comparison-model implementations.
- `encoder/`: BERT sentence encoder.
- `dataset/`: FewRel and FS-TACRED data loaders.
- `toolkit/`: training and evaluation frameworks.
- `scripts/`: baseline reproduction scripts.
- `data/`: dataset placement instructions and relation description pools.

