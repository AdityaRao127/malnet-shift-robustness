# malnet-shift-robustness

This project reproduces parts of Tran et al. and extends their approach by allowing the model to explicitly use missing feature information instead of relying only on zero-filled inputs.

All experiments were run on Google Colab Pro using a T4 GPU.

## Overview

The goal of this project is to study how graph-based malware classifiers perform under distribution shift. I start with a baseline GCN model and gradually improve the feature representation, then evaluate how each version performs on both standard and shifted data.

The main addition in this project is a gating model that takes a missingness mask as input and learns how to adjust the graph representation based on which features are available.

## Notebooks

The notebooks should be run in order since each stage builds on the previous one.

[assignment3_baseline.ipynb](notebooks/assignment3_baseline.ipynb)
This is the starting point. It uses a vanilla GCN with one-hot degree features and reaches about 75.1% accuracy on the MalNet-Tiny test set.

[phase1_structural.ipynb](notebooks/phase1_structural.ipynb)
Replaces the one-hot degree input with structural features such as Local Degree Profile, clustering, PageRank, and BFS depth.

[phase2_enriched.ipynb](notebooks/phase2_enriched.ipynb)
Adds additional graph-derived features and simulates missingness during training. Two approaches are tested: zero-fill and prune.

[phase3_gating.ipynb](notebooks/phase3_gating.ipynb)
Introduces the gating model. The model receives a missingness mask and learns how to adjust the embedding based on which feature groups are present.

[phase4_shifted.ipynb](notebooks/phase4_shifted.ipynb)
Evaluates all models on the MalNet-Tiny-Common shifted split and computes the robustness gap.

## Running the code

Open any notebook in Google Colab.
Change the runtime to a T4 GPU.
Run all cells.

The setup installs the required dependencies and clones the repository.
Results are saved in the results folder.

## Project structure

[notebooks](notebooks/)
Contains all experiment notebooks.

[src](src/)
Contains shared code for models, training, feature generation, and missingness simulation.

[scripts](scripts/)
Contains helper scripts for running experiments and generating notebooks.

[tests](tests/)
Contains a basic test to verify imports and setup.

## References

See paper.
