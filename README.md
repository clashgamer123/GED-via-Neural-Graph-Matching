# GED-via-Neural-Graph-Matching (CS 768)

An extension of the **GedGNN** architecture (VLDB 2023) for neural Graph Edit Distance (GED) approximation. This project introduces a **configurable Graph Transformer encoder**, enables **GPU acceleration**, and provides a comprehensive **ablation study** on depth, attention heads, and residual connections across AIDS, Linux, and IMDB datasets.

---

## 🚀 Quick Start (Demo Mode)
To verify the setup and see a "smoke test" run (1 epoch on a small subset of data):

```bash
# Navigate to the active experiment directory
cd "project_files/experiments/Overall Performance/"

# Run the Transformer-based GedGNN in demo mode
python src/main.py --model-name GedGNN --dataset AIDS --gnn-operator transformer --demo --model-train 1 --model-epoch-end 1
```

---

## 🛠 Project Overview & Contributions
The core of this project is extending the GedGNN pipeline (Encoder → Differentiable Matching → k-best Post-processing) by replacing the local GIN/GCN encoders with a more expressive Transformer-based backbone.

### Key Modifications
*   **Graph Transformer Encoder**: Integrated `TransformerConv` with multi-head attention.
*   **Hyperparameter Control**: Added CLI flags for `--heads`, `--num-layers`, and `--residual`.
*   **GPU Optimization**: Fixed hardcoded CPU flags and tensor device synchronization bugs.
*   **Bug Fixes**: Patched an `IndexError` in the precision-at-k (`cal_pk`) metric calculation.

---

## 📁 Repository Structure
*   `project_files/experiments/Overall Performance/src/`: **Active Source Code.**
*   `project_files/experiments/Overall Performance/result/`: Contains `SUMMARY.md` with final metrics.
*   `IMPLEMENTATION_NOTES.md`: Detailed technical rationale and design decisions.
*   `VIVA_PREP.md`: Key questions and "cheat sheet" for project defense.

---

## 📊 Experimental Results (Highlights)
We ran a 24-cell ablation study comparing GIN/GCN baselines against various Transformer configurations.

*   **IMDB (Large Graphs)**: The Transformer encoder (L=4) achieved a **2.2x improvement in MAE** (4.42 vs 9.75) over the GIN baseline.
*   **AIDS (Small Graphs)**: While absolute error was similar, the Transformer significantly improved **ranking metrics** (Spearman ρ jumped from 0.11 to 0.68).
*   **Depth scaling**: Our findings confirm that encoder depth should scale with graph size; deeper models improved IMDB results but caused over-smoothing on smaller AIDS graphs.

---

## ⚙️ Detailed Running Instructions
All commands should be run from `project_files/experiments/Overall Performance/`.

### 1. Training a Model
To train the Graph Transformer for 20 epochs on AIDS:
```bash
python src/main.py --model-name GedGNN --dataset AIDS --gnn-operator transformer --heads 4 --num-layers 3 --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```

### 2. Evaluation / Post-processing
To run the k-best matching algorithm (k=1000) using a saved checkpoint:
```bash
python src/main.py --model-name GedGNN --dataset AIDS --gnn-operator transformer --model-train 0 --model-epoch-start 20 --postk 1000
```

---

## 📦 Requirements
*   Python 3.8+
*   PyTorch 1.8.2 / PyTorch Geometric 2.0.4
*   DGL 0.7.0
*   NetworkX / Scipy / Matplotlib
