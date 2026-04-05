# malnet-shift-robustness

Reproducing parts of Tran et al. (https://arxiv.org/abs/2508.06734) and add a change where the model knows which feature groups are missing instead of just zero filling them.

Runs on Google Colab Pro with a T4 GPU.

## Phase notebooks

Run them in order, cause each phase builds on the previous.

- `assignment3_baseline.ipynb`: vanilla GCN with one hot degree features. Got 75.1% on MalNetTiny test. This is the starting point.
- `phase1_structural.ipynb`: swaps the one hot degree for LDP plus a few extra graph features (clustering, PageRank, BFS depth). Same model, just better inputs.
- `phase2_enriched.ipynb`: adds more graph derived features on top of phase 1 and randomly drops a chunk of them at train time. Tries two ways of handling the missing chunk (zero fill and prune).
- `phase3_gating.ipynb`: this is the new bit. The model gets the missingness mask as an extra input and learns to gate the embedding based on what is there.
- `phase4_shifted.ipynb`: pulls the Common shifted split from HuggingFace and runs all four models on it. Computes the robustness gap.

## How to run on Colab

1. Open a notebook.
2. In the menu pick Runtime, then Change runtime type, then T4 GPU.
3. Select Run All.

The first cell clones this repo. The pip install cell pins torch-geometric to 2.7.0 to avoid weird API drift. Each notebook writes its outputs to `results/`.

## What's in folders

- `notebooks/` is where the runnable stuff lives.
- `src/` has the shared model code, training loop, feature transforms, missingness simulation, HF loader.
- `scripts/` has a CLI runner and a few small helpers for generating the notebooks.
- `tests/` has a smoke test for the imports.

## Papers I'm building on

- Tran et al., https://arxiv.org/abs/2508.06734
- Freitas et al. (MalNet dataset), https://openreview.net/pdf?id=1xDTDk3XPW
- Kipf and Welling (GCN), https://arxiv.org/abs/1609.02907
- Cai and Wang (LDP features), https://arxiv.org/abs/2003.00982
- Tran et al.'s precomputed splits on HuggingFace: https://huggingface.co/datasets/nntvu/MalNet-Tiny-Features
