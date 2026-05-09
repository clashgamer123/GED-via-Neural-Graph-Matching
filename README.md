# GED-via-Neural-Graph-Matching (CS 768)

**Course:** CS 768 – Graph Representation Learning  
**Project:** Neural Graph Matching for Graph Edit Distance (GED) Approximation

---

## Project Overview

This repository empirically investigates and extends **GedGNN**, a state-of-the-art framework for neural Graph Edit Distance approximation. 

> Piao, C., et al. (2023). *Computing Graph Edit Distance via Neural Graph Matching.* PVLDB 2023.

The key contribution of this project is the integration of a **Configurable Graph Transformer encoder** to replace standard local message-passing backends (GIN/GCN). We validate that **multi-head neighborhood attention** and deeper architectures significantly alleviate the performance bottlenecks found in traditional GNNs when processing larger, more complex graphs.

---

## Repository Structure

```text
root/
├── README.md                          # Comprehensive project guide
├── IMPLEMENTATION_NOTES.md            # Technical rationale & design decisions
├── VIVA_PREP.md                       # Defense Q&A and study guide
│
└── project_files/
    ├── requirements.txt               # Dependencies (PyTorch, PyG, DGL)
    └── experiments/
        └── Overall Performance/
            ├── result/                # Aggregated metrics (SUMMARY.md) & logs
            ├── model_save/            # Checkpoint storage
            └── src/                   # Active Implementation
                ├── models.py          # GedGNN with configurable Transformer backend
                ├── trainer.py         # Training loop, GPU synchronization & metrics
                ├── param_parser.py    # Hyperparameter CLI interface
                ├── GedMatrix.py       # Differentiable Cost & Mapping modules
                └── kbest_matching_with_lb.py # Murty’s k-best Hungarian post-processing
```

---

## Setup

### 1. Environment Configuration
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r project_files/requirements.txt
```

### 2. Core Dependencies
- **PyTorch / PyTorch Geometric:** Neural matching and `TransformerConv` implementation.
- **DGL:** Graph data structures and legacy baseline aggregations.
- **NetworkX / Scipy:** Graph manipulation and evaluation utilities.

---

## Running the Experiments

All commands must be executed from `project_files/experiments/Overall Performance/`.

### Phase 1: Quick Demo (Smoke Test)
Run a single epoch on a subset of the AIDS dataset to verify the Transformer pipeline:
```bash
python src/main.py --model-name GedGNN --dataset AIDS --gnn-operator transformer --demo --model-train 1 --model-epoch-end 1
```

### Phase 2: Full Training Sweep
Train the Graph Transformer (4 heads, 3 layers) on the IMDB dataset for 20 epochs:
```bash
python src/main.py --model-name GedGNN --dataset IMDB --gnn-operator transformer --heads 4 --num-layers 3 --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```

### Phase 3: Post-Processing & Refinement
Execute Murty’s $k$-best matching algorithm using a pre-trained checkpoint to refine GED estimates:
```bash
python src/main.py --model-name GedGNN --dataset IMDB --gnn-operator transformer --model-train 0 --model-epoch-start 20 --postk 1000
```

---

## Key Experimental Results

| Dataset | Metric | GIN (Baseline) | Transformer (Ours) | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| **IMDB** | MAE | 9.75 | **4.42** | **2.2x Lower Error** |
| **AIDS** | Spearman $\rho$ | 0.11 | **0.68** | **6.1x Better Ranking** |

**Headline Findings:**
- **Depth Scaling:** For larger ego-networks (IMDB), increasing encoder depth is the primary driver for accuracy, with MAE dropping monotonically from 13.9 to 4.4 across 4 layers.
- **Attention Synergy:** In molecular retrieval (AIDS), attention-based mixing significantly improves ranking metrics ($\rho$), even when pointwise error (MAE) remains comparable to GIN.

---

## Code Reference

### `models.py`
- `GedGNN`: Core architecture implementing differentiable matching.
- `Transformer branch`: Utilizes `out_dim // heads` logic to maintain constant parameter counts across head ablations.
- `Residual Projections`: Employs learned `Linear` mappings for dimension alignment in the residual path.

### `trainer.py`
- `fit()`: Orchestrates the dual-loss objective (Pointwise Value Loss + Structural Mapping Supervision).
- `cal_pk()`: Patched metric calculation to ensure stability during small-scale and demo evaluations.
- `self.use_gpu`: Robust device detection for seamless CPU/GPU transitions.

### `kbest_matching_with_lb.py`
- `KBestMSolver`: Implementation of Murty's algorithm with lower-bound pruning for exact GED refinement.

---

## References

1. Piao, C., et al. (2023). *Computing Graph Edit Distance via Neural Graph Matching.* PVLDB.
2. Shi, Y., et al. (2021). *Masked Label Prediction: Unified Message Passing Model for Semi-Supervised Classification.* (TransformerConv implementation).
3. Xu, K., et al. (2019). *How Powerful are Graph Neural Networks?* ICLR. (GIN baseline).
