# Implementation Notes — GedGNN with Graph Transformer Encoder

**Course:** CS768
**Base paper:** Piao et al., *"Computing Graph Edit Distance via Neural Graph Matching"* (GedGNN), VLDB 2023.
**Repository:** `GED-via-Neural-Graph-Matching/project_files/experiments/Overall Performance/`

---

## 1. One-line project description (for the submission form)

> *"Extension and ablation study of GedGNN: we replace its GIN encoder with a Graph Transformer (multi-head `TransformerConv` with LayerNorm and optional residual connections), make encoder depth and attention heads first-class hyperparameters, and ablate them along with the k-best post-processing budget against the GIN and GCN baselines on AIDS-700."*

---

## 2. What GedGNN does (background, for the report)

GedGNN computes Graph Edit Distance (GED) between two graphs `G1, G2` in three stages:

1. **Encoder.** A shared GNN (originally GIN) maps each node of each graph to a `d`-dim embedding. The embeddings are computed independently for the two graphs.
2. **Differentiable matching.** Two `GedMatrixModule`s produce (i) a **cost matrix** `C ∈ R^{n1×n2}` and (ii) a **mapping logits matrix** `M ∈ R^{n1×n2}`. Softmax over rows of `M` gives a soft node-to-node correspondence; element-wise multiplied with `C` and summed gives the differentiable GED estimate. A scalar **bias** is added from a Neural Tensor Network (NTN) over the pooled graph embeddings.
3. **Post-processing.** At inference time, the soft mapping is turned into a hard one with **k-best Hungarian matching** (`kbest_matching_with_lb.py`) — they enumerate the top-k assignments and pick the one that minimizes the actual GED, giving the final discrete prediction.

The encoder is the only part of the pipeline that operates on a single graph in isolation, so it is the natural place to inject more expressive architectures.

---

## 3. Motivation for the extension

GIN is provably as expressive as the 1-WL test and is therefore strong for graph isomorphism, but it is purely *local* (a node sees only its 1-hop multiset at each layer). Two limitations matter for GED:

- **Long-range substructures.** GED depends on global structure (graph diameter, connectivity, near-symmetries). A k-layer GIN can only see k-hop neighborhoods.
- **Anisotropic / attention-style mixing.** When matching two graphs, some neighbors are more diagnostic than others. A learned attention mechanism over neighbors gives the encoder the capacity to upweight them.

Graph Transformers (here PyG's `TransformerConv` — Shi et al., 2021) replace GIN's sum aggregation with **multi-head dot-product attention over neighbors** (with edge bias). They have repeatedly outperformed message-passing GNNs on tasks where global structure matters. We therefore plug a Graph Transformer in as the encoder of GedGNN and study the resulting design space.

---

## 4. Concrete code changes

All edits are confined to two files in `project_files/experiments/Overall Performance/src/`. The default behavior of the original code (GIN/GCN) is unchanged; the new flags only affect the `transformer` branch.

### 4.1 `param_parser.py` — three new CLI flags

```python
parser.add_argument("--heads",        type=int, default=4,
                    help="Number of attention heads for TransformerConv.")
parser.add_argument("--num-layers",   type=int, default=3,
                    help="Number of GNN encoder layers (transformer only).")
parser.add_argument("--residual",     action="store_true", default=False,
                    help="Enable residual connections between Transformer layers.")
```

These convert previously hard-coded constants (4 heads, 3 layers, no residual) into hyperparameters that can be swept from the command line.

### 4.2 `models.py` — refactored `GedGNN` Transformer encoder

**Before (original code):** three hard-coded `TransformerConv` layers with `heads=4`, two `LayerNorm`s, no residual:

```python
self.convolution_1 = TransformerConv(num_labels,  filters_1 // 4, heads=4, ...)
self.convolution_2 = TransformerConv(filters_1,   filters_2 // 4, heads=4, ...)
self.convolution_3 = TransformerConv(filters_2,   filters_3 // 4, heads=4, ...)
self.norm_1 = LayerNorm(filters_1)
self.norm_2 = LayerNorm(filters_2)
```

**After:** a configurable `ModuleList` of layers, with per-layer LayerNorm and optional residual projections:

```python
heads = self.args.heads
num_layers = self.args.num_layers
if num_layers == 3:
    dims = [num_labels, filters_1, filters_2, filters_3]   # backward-compat
else:
    dims = [num_labels] + [filters_1] * (num_layers - 1) + [filters_3]

self.tf_convs          = nn.ModuleList()
self.tf_norms          = nn.ModuleList()
self.tf_residual_proj  = nn.ModuleList()
for i in range(num_layers):
    in_dim, out_dim = dims[i], dims[i+1]
    assert out_dim % heads == 0
    self.tf_convs.append(TransformerConv(in_dim, out_dim // heads,
                                         heads=heads, dropout=self.args.dropout))
    if i < num_layers - 1:
        self.tf_norms.append(LayerNorm(out_dim))
        if self.args.residual:
            self.tf_residual_proj.append(
                nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            )
```

`convolutional_pass` was rewritten to iterate this list:

```python
if self.args.gnn_operator == 'transformer':
    for i, conv in enumerate(self.tf_convs):
        prev = features
        h = conv(features, edge_index)
        if i < len(self.tf_convs) - 1:
            h = self.tf_norms[i](h)
            h = F.elu(h)
            if self.args.residual:
                h = h + self.tf_residual_proj[i](prev)
            h = F.dropout(h, p=self.args.dropout, training=self.training)
        features = h
    return features
```

**Design notes (the "why" you can quote in the report):**

- **`out_dim // heads`.** `TransformerConv` outputs `heads × out_per_head` features. We keep total output width fixed at `filters_i` so that the downstream `GedMatrixModule` and NTN see the same dimensionality regardless of `--heads`. This is the only valid way to do an apples-to-apples heads ablation; varying heads while keeping per-head width constant would change the parameter count and the cost-matrix dimensionality at the same time, confounding the two.
- **LayerNorm + ELU between layers, no activation on the last layer.** Matches the standard Graph Transformer block (Pre-LN-style would also work, but Post-LN keeps the diff minimal vs. the original code). The last layer is left un-normalized so its scale matches what the `costMatrix` / `mapMatrix` modules were trained against in the GIN baseline.
- **Residual with linear projection on dim mismatch.** The first transition (`num_labels → filters_1`) and the last transition (`filters_2 → filters_3`) change width, so a vanilla residual is not type-correct; a learned `Linear` is the standard fix. We made it `nn.Identity()` whenever dims happen to match (e.g., when `num_layers > 3`, intermediate transitions are `filters_1 → filters_1`).
- **Backward compatibility.** When `num_layers == 3`, we reproduce the exact `[num_labels → filters_1 → filters_2 → filters_3]` schedule of the original code, so the new code is a strict generalization of the old.

**What we deliberately did *not* change** (and why):
- The `costMatrix`, `mapMatrix`, NTN bias head, Sinkhorn / softmax, k-best post-processing, loss function, and data loader are all untouched. This isolates the encoder as the only independent variable in our ablations — any difference in metrics is attributable to the encoder change.

---

## 5. Experimental protocol

### 5.1 Dataset
AIDS-700 (the dataset shipped under `json_data/AIDS/`), with the train/val/test split provided by the repo and ground-truth GEDs from `TaGED.json`. 20 training epochs (the value in the original `arg.txt`).

### 5.2 Metrics (already implemented in `trainer.py`)
- **MAE** and **MSE** of predicted GED vs. ground truth.
- **Spearman ρ** and **Kendall τ** of the induced ranking of test pairs.
- **Precision@10**, **Precision@20** for retrieving the closest graphs.
- **Inference time per pair** (matters for the post-processing ablation).

### 5.3 Variants we ran

| Variant            | Encoder         | Flags                                                    | Purpose                              |
|--------------------|-----------------|----------------------------------------------------------|--------------------------------------|
| Baseline-GIN       | GIN             | `--gnn-operator gin`                                     | Reference (paper's setup)            |
| Baseline-GCN       | GCN             | `--gnn-operator gcn`                                     | Reference (weaker baseline)          |
| GT-default         | GraphTransformer| `--gnn-operator transformer --heads 4 --num-layers 3`    | Encoder swap, like-for-like          |
| GT-heads-{1,2,8}   | GraphTransformer| `--heads {1,2,8}`                                        | Heads ablation                        |
| GT-layers-{2,4}    | GraphTransformer| `--num-layers {2,4}`                                     | Depth ablation                        |
| GT-residual        | GraphTransformer| `--residual`                                             | Architectural tweak                   |
| GT-postk-{10,..,10000} | GraphTransformer | `--postk {…}` (no retrain — eval only)              | Post-processing budget tradeoff       |

### 5.4 Commands

Training (one for each variant):
```bash
python src/main.py --model-name GedGNN --dataset AIDS \
    --gnn-operator transformer --heads 4 --num-layers 3 \
    --model-epoch-start 0 --model-epoch-end 20 --model-train 1
```

Post-processing-only re-evaluation (re-uses the saved checkpoint, sweeps `postk`):
```bash
python src/main.py --model-name GedGNN --dataset AIDS \
    --gnn-operator transformer --heads 4 \
    --model-epoch-start 20 --model-epoch-end 20 --model-train 0 \
    --postk 1000
```

Results land in `result/`; each run produces a CSV-like text file with the metrics above.

---

## 6. Hypotheses to discuss in the report

1. **GT vs GIN.** We expect GT-default to match or modestly beat Baseline-GIN on MAE/MSE, with a larger improvement on Spearman ρ (because attention helps order-sensitive metrics).
2. **Heads.** Performance should be flat / mildly U-shaped: 1 head ≈ a learned weighted GCN (under-expressive); 8 heads partition the 32-dim final embedding into 4-dim per-head slices that may be too narrow. 2–4 heads is usually the sweet spot.
3. **Depth.** More layers help GT more than they help GIN (bigger receptive field amplifies the value of attention). However, AIDS graphs are tiny (~10 nodes), so returns diminish past 3 layers.
4. **Residual.** Helps mainly at higher depth; for `num_layers=3` the effect is small but should at least not hurt.
5. **postk tradeoff.** MAE should drop monotonically with `postk` and saturate around `postk≈1000`; runtime grows roughly linearly. The plot of MAE vs. wall-clock per pair is the headline figure.

---

## 7. Limitations / honest disclosure (good to include)

- **No new dataset.** All experiments are on AIDS-700; cross-dataset generalization to LINUX/IMDB was out of scope due to the submission timeline.
- **No new loss / matching scheme.** Our contribution is on the encoder side only; the differentiable matching and k-best post-processing are unchanged from GedGNN.
- **No positional encodings.** Pure Graph Transformers usually benefit from Laplacian / RWPE positional encodings. We did not add them in this round to keep the comparison clean (encoder-architecture only); this is the most natural follow-up.
- **TransformerConv ≠ "full" Graph Transformer.** `TransformerConv` is a *local* attention layer — attention is only over 1-hop neighbors, not all pairs. This is the right baseline for fairness against GIN/GCN but worth flagging.

---

## 8. Files modified

| File                                                                                          | Change                                                       |
|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------|
| `project_files/experiments/Overall Performance/src/param_parser.py`                           | +3 CLI flags: `--heads`, `--num-layers`, `--residual`         |
| `project_files/experiments/Overall Performance/src/models.py` (`GedGNN` class)                | Refactored transformer encoder to a configurable `ModuleList` with optional residual projections |

No changes to `trainer.py`, `utils.py`, `kbest_matching_with_lb.py`, `GedMatrix.py`, or `layers.py`.

---

## 9. Reproducing a single run end-to-end

```bash
cd project_files/experiments/Overall\ Performance/

# Train (20 epochs, ~few minutes on a GPU on AIDS)
python src/main.py --model-name GedGNN --dataset AIDS \
    --gnn-operator transformer --heads 4 --num-layers 3 --residual \
    --model-epoch-start 0 --model-epoch-end 20 --model-train 1

# Evaluate (k-best post-processing with k=1000)
python src/main.py --model-name GedGNN --dataset AIDS \
    --gnn-operator transformer --heads 4 --num-layers 3 --residual \
    --model-epoch-start 20 --model-epoch-end 20 --model-train 0 --postk 1000
```

Open the file written under `result/` to read off MAE, MSE, ρ, τ, p@10, p@20, and per-pair time.
