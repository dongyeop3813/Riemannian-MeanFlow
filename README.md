# Riemannian Mean Flow (RMF)

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

<p align="center">
  <img src="assets/method.png" alt="Method overview" width="600"/>
</p>

Official JAX implementation of **Riemannian Mean Flow** for flow-map learning on Riemannian manifolds. This repository reproduces the **toy**, **Earth**, and **DNA** experiments from the paper *Riemannian Mean Flow*. Protein experiments are provided separately at [https://xxx](https://xxx).

## Implemented experiments

| Experiment   | Manifold              | Data                                                                 | Description                          |
| ------------ | --------------------- | -------------------------------------------------------------------- | ------------------------------------ |
| **Toy-helix** | Sphere (Sⁿ)           | Synthetic helix on the hypersphere Sⁿ                               | Validation of flow-map learning on Sⁿ    |
| **Earth**    | Sphere (S²)           | Geographic events (earthquake, volcano, fire, flood) as points on S² | Density estimation on the Earth      |
| **DNA**      | Product of simplices  | DNA promoter sequences (1024×4)                                     | Conditional generation on sequences  |

## Installation

Requires Python ≥3.10. The project uses **uv** for environment and dependency management:

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies from project root
uv sync
```

## Data preparation

Experiments use the following data; prepare each as below before running.

### Earth

Place geographic event data (CSV with lat/lon) under `data/earth_data/`. For download and setup, follow the instructions in [facebookresearch/riemannian-fm](https://github.com/facebookresearch/riemannian-fm).

### DNA promoter

Download the dataset from [Zenodo (7943307)](https://zenodo.org/records/7943307) and place it in `data/dna_promoter`. The dataset is from **Stark et al.** — *Dirichlet Flow Matching with Applications to DNA Sequence Design* (2024).

## Quick start

### 1. Toy helix (Sⁿ)

Synthetic helix on the sphere; supports Eulerian (RMF), Lagrangian (LMF), and Semigroup (SMF) formulations:

```bash
uv run main.py experiment=rmf_toy_helix   # Eulerian RMF
uv run main.py experiment=lmf_toy_helix   # Lagrangian LMF
uv run main.py experiment=smf_toy_helix   # Semigroup SMF
```

### 2. Earth

Density estimation on the sphere for geographic event data. Place CSV data under `data/earth_data/` and select the dataset:

```bash
uv run main.py experiment=rmf_earth data=earthsquake   # earthquake (default)
uv run main.py experiment=rmf_earth data=volcano
uv run main.py experiment=rmf_earth data=fire
uv run main.py experiment=rmf_earth data=flood
```

### 3. DNA promoter

Conditional flow on DNA promoter sequences (length 1024, 4 nucleotides). SEI evaluation is enabled when `eval.eval_sei: true` in the task config.

```bash
uv run main.py experiment=rmf_promoter
```

## Project structure

```
config/
├── experiment/    # rmf_*, lmf_*, smf_* per task
├── task/          # earth, promoter_dna, toy_helix
├── data/          # dataset configs (earthsquake, volcano, dna_promoter, ...)
├── model/         # flow model and network configs
├── optim/         # optimizer and LR schedule
└── time_sampler/  # time sampling for flow matching

src/
├── manifold/           # Sphere, Simplex, ProductManifold
├── experiments/       # MeanFlowTrainer, flowmap_learning (losses, GFM, ...)
├── probability_path/  # geodesic and probability paths
├── data/              # toy_helix, earth, dna_promoter datasets
├── model/             # flow models (MLP, promoter_model)
├── eval/              # MMD, SEI, FBD, k-mer, toy/earth eval
└── utils/             # training, checkpointing, logging
```

Configuration is managed by **Hydra**; override options from the CLI, e.g. `uv run main.py experiment=rmf_toy_helix batch_size=512`.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{riemannian_mean_flow,
  title   = {Riemannian Mean Flow},
  author  = {},
  journal = {},
  year    = {},
}
```