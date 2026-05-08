# Code Walkthrough — GedGNN Project

This is a file-by-file tour of the codebase. Read this **before** the viva prep — it gives you the mental model. Goal: by the end of this doc, you can answer "what does file X do?" for every file we touched or used.

All paths below are relative to `project_files/experiments/Overall Performance/`. The active code lives in `src/`.

---

## 0. Entry point — `src/main.py` (40 lines, trivial)

```python
args = parameter_parser()       # parse CLI flags
trainer = Trainer(args)         # build trainer (also builds model + loads data)
if args.model_epoch_start > 0:
    trainer.load(args.model_epoch_start)   # resume from checkpoint

if args.model_train == 1:
    for epoch in range(args.model_epoch_start, args.model_epoch_end):
        trainer.cur_epoch = epoch
        trainer.fit()           # one training epoch
        trainer.save(epoch + 1) # checkpoint
        trainer.score('test')   # eval
else:
    trainer.cur_epoch = args.model_epoch_start
    trainer.score('test', test_k=0)  # eval-only mode
```

**Mental model:** main.py is a thin shell. All the logic is in `Trainer`. You call it with `--model-train 1` to train+eval, or `--model-train 0` to eval an existing checkpoint. The two epoch flags `--model-epoch-start/end` define the outer training loop.

---

## 1. CLI flags — `src/param_parser.py`

A vanilla argparse setup. Important flags grouped by purpose:

**Architecture**:
- `filters_1=128, filters_2=64, filters_3=32` — encoder layer widths.
- `tensor_neurons=16` — output dim of the NTN bias module.
- `bottle_neck_neurons=16, _2=8, _3=4` — FC head after NTN.
- `hidden_dim=16` — MLP hidden dim inside `GedMatrixModule`.
- `dropout=0.5`.
- `gnn_operator ∈ {gcn, gin, transformer}` — which encoder.
- **`heads=4` (we added)** — attention heads.
- **`num_layers=3` (we added)** — encoder depth.
- **`residual` (we added)** — toggle residuals.

**Training**:
- `epochs, batch_size=128, learning_rate=0.001, weight_decay=5e-4`.
- `loss_weight=1.0` — weight of value loss vs. mapping loss in GedGNN.
- `target_mode ∈ {linear, exp}` — how the predicted similarity score is converted back to a GED estimate (see Q below).

**Data / eval**:
- `dataset ∈ {AIDS, Linux, IMDB}`.
- `model_name ∈ {SimGNN, GPN, GedGNN, TaGSim}` — we use `GedGNN`.
- `graph_pair_mode='combine'`, `num_delta_graphs=100`, `num_testing_graphs=100`.
- `demo` — shrinks dataset to 30/40/50 graphs for fast smoke tests.
- `postk=1000` — k for k-best Hungarian post-processing.

> **`target_mode` explained:** the model emits a sigmoid score `s ∈ [0,1]`. With `target_mode=linear`, `pre_ged = s * hb` where `hb` is a graph-pair-specific upper bound. With `target_mode=exp`, `pre_ged = -log(s) * avg_v` (matches the SimGNN convention where small distance → high similarity ≈ 1). This is just a parameterization of how the network expresses GED.

---

## 2. Data pipeline — `src/utils.py` + `Trainer.load_data` + `Trainer.transfer_data_to_torch`

### `utils.py`
Helper functions for reading the dataset directory structure:
- `iterate_get_graphs(dir, fmt)` — read all graphs from `dir` in format `gexf|json|onehot|anchor`.
- `load_all_graphs(path, dataset)` — returns `(train_num, val_num, test_num, graphs)` where `graphs` is a list of dicts.
- `load_labels(path, dataset)` — for AIDS only, loads one-hot label features (29 classes).
- `load_ged(ged_dict, ...)` — populates `ged_dict[(id1,id2)] = ((value, nc, in, ie), best_mappings_list)` from `TaGED.json`.

### `Trainer.load_data` (`trainer.py:113`)
- Calls `load_all_graphs` for the chosen dataset.
- For AIDS: `number_of_labels=29`, features are one-hot per node.
- For Linux / IMDB: `number_of_labels=1`, features are constant `[2.0]` per node (so the feature matrix is just a column of 2's). The model has nothing to do with the features — all signal comes from structure.
- Calls `load_ged` to build the GT GED dictionary.

### `Trainer.transfer_data_to_torch` (`trainer.py:144`)
Converts the loaded graphs into tensors that live on GPU:
- Builds `self.edge_index[i]`: edge list with reverse edges + self-loops added (so the GNN message-passing covers both directions and the node itself). Shape `(2, |E|)`.
- Builds `self.features[i]`: the node feature tensor.
- Builds the `n × n` `mapping[i][j]` matrix (used as ground-truth supervision for the soft mapping). It's a 0/1 matrix where `mapping[i][j][u][v] = 1` iff some optimal node mapping sends node `u` of graph `i` to node `v` of graph `j`.
- Builds `self.ged[i][j]` — the (value, nc, in, ie) GT GED for each pair.

### `Trainer.init_graph_pairs` (`trainer.py:273`)
Where `--demo` actually has effect:
```python
if self.args.demo:
    train_num = 30
    val_num = 40
    test_num = 50
    self.args.epochs = 1
```
Then it iterates `(i, j)` pairs and produces three lists: `training_graphs`, `val_graphs`, `testing_graphs`. Each entry is a triple `(pair_type, i, j)` where `pair_type ∈ {0=normal pair from dataset, 1=delta graph synthetic pair, 2=other-set pair}`.

The "delta graph" mechanism (`Trainer.delta_graph`, `trainer.py:197`) is a synthetic data augmentation: takes a graph `g`, applies a random small set of edits to produce `g'`, and returns `(g', mapping_g_to_g', GED)`. So for each graph in training, you get 100 synthetic neighbors to train against. This is how they get enough training signal from a small dataset.

### `Trainer.pack_graph_pair` (`trainer.py:354`)
Per-batch data wrangling. Given a `(pair_type, i, j)`, it returns a dict containing:
- `edge_index_1, edge_index_2`: edge tensors.
- `features_1, features_2`: node feature tensors.
- `n1, n2`: node counts.
- `ta_ged`: ground-truth GED tuple.
- `target`, `mapping`: supervision targets.
- `avg_v, hb`: normalization constants for `target_mode`.

The model's `forward(data)` reads exactly this dict.

---

## 3. The encoder — `src/models.py` `GedGNN` class (THE FILE WE MODIFIED)

### Class layout (post-our-changes)

```
class GedGNN(torch.nn.Module):
    def __init__(args, num_labels):
        self.args = args
        self.number_labels = num_labels
        self.setup_layers()       # builds all submodules

    def setup_layers(self):
        # === ENCODER (this is where we added the new transformer code) ===
        if gnn_operator == 'gcn':       3 hard-coded GCNConv layers
        elif gnn_operator == 'gin':     3 hard-coded GINConv layers (each is a small MLP)
        elif gnn_operator == 'transformer':
            # OUR CODE — configurable ModuleList of TransformerConv
            heads, num_layers, residual = ...
            dims = [num_labels, 128, 64, 32]   if num_layers == 3
                   [num_labels, 128, ..., 128, 32]  otherwise
            self.tf_convs         = ModuleList([TransformerConv(...) for each layer])
            self.tf_norms         = ModuleList([LayerNorm(...) for non-final layers])
            self.tf_residual_proj = ModuleList([Linear or Identity for non-final layers])

        # === MATCHING MATRICES (unchanged) ===
        self.mapMatrix  = GedMatrixModule(filters_3=32, hidden_dim=16)
        self.costMatrix = GedMatrixModule(filters_3=32, hidden_dim=16)

        # === BIAS HEAD (NTN + FC stack, unchanged) ===
        self.attention      = AttentionModule(args)
        self.tensor_network = TensorNetworkModule(args)
        self.fully_connected_first  = Linear(tensor_neurons=16, bottle_neck_neurons=16)
        self.fully_connected_second = Linear(16, 8)
        self.fully_connected_third  = Linear(8, 4)
        self.scoring_layer          = Linear(4, 1)
```

### `convolutional_pass(edge_index, features)`
Runs the chosen encoder. After our refactor, the transformer branch iterates `tf_convs`:

```python
if gnn_operator == 'transformer':
    for i, conv in enumerate(self.tf_convs):
        prev = features
        h = conv(features, edge_index)        # multi-head attention over 1-hop neighbors
        if i < num_layers - 1:                # all but last
            h = self.tf_norms[i](h)
            h = F.elu(h)
            if residual:
                h = h + self.tf_residual_proj[i](prev)
            h = F.dropout(h, p=0.5)
        features = h
    return features                            # shape (n, 32)
```

### `get_bias_value(H1, H2)`
Computes the scalar GED bias from pooled embeddings:
```python
p1 = self.attention(H1)              # (32,1) — global graph descriptor of G1
p2 = self.attention(H2)              # (32,1) — same for G2
s  = self.tensor_network(p1, p2)     # (16,)  — NTN similarity vector
s  = ReLU(self.fully_connected_first(s))     # 16 → 16
s  = ReLU(self.fully_connected_second(s))    # 16 → 8
s  = ReLU(self.fully_connected_third(s))     # 8 → 4
return self.scoring_layer(s).view(-1)        # 4 → 1
```

### `forward(data)` — the punchline
```python
H1 = self.convolutional_pass(edge_index_1, features_1)  # (n1, 32)
H2 = self.convolutional_pass(edge_index_2, features_2)  # (n2, 32)

cost_matrix = self.costMatrix(H1, H2)   # (n1, n2)
map_matrix  = self.mapMatrix (H1, H2)   # (n1, n2)

soft_matrix = softmax(map_matrix, dim=1) * cost_matrix   # (n1, n2)
bias_value  = self.get_bias_value(H1, H2)                # scalar
score       = sigmoid(soft_matrix.sum() + bias_value)    # scalar in [0,1]

# convert back to GED units
if target_mode == 'exp':    pre_ged = -log(score) * avg_v
elif target_mode == 'linear': pre_ged = score * hb

return score, pre_ged.item(), map_matrix
```

**This is the heart of GedGNN.** Read it three times until it sticks. The model produces a soft node correspondence + per-pair edit cost; element-wise multiply and sum gives the predicted GED. The bias term lets the model add a global correction.

---

## 4. The matching matrix — `src/GedMatrix.py`

### `GedMatrixModule(d, k)`
Used twice in GedGNN: once as `costMatrix`, once as `mapMatrix`. Both have `d=32` (final encoder width) and `k=16` (hidden_dim). Pure 4D-bilinear-then-MLP interaction:

```python
W ∈ R^(k, d, d)            # learnable, k=16 weight matrices of shape d×d
forward(H1: (n1,d), H2: (n2,d)) -> (n1, n2):
    x = H1 @ W            # (n1, d) @ (k, d, d) -> (k, n1, d) via broadcast
    x = x @ H2.t()        # -> (k, n1, n2)
    x = x.reshape(k, -1).t()    # -> (n1*n2, k)
    x = MLP(x)            # k -> 2k -> k -> 1, three Linear+ReLU layers
    return x.reshape(n1, n2)
```

So each entry `(i,j)` of the output is a learned function of the bilinear interaction `H1[i] · W_q · H2[j]` for q=1..k.

`SimpleMatrixModule(k)` is a stripped-down version (element-wise product instead of bilinear), unused in GedGNN's default config — there's a commented-out alternative line `# self.costMatrix = SimpleMatrixModule(filters_3)` in `models.py`.

---

## 5. Pooling + NTN — `src/layers.py`

### `AttentionModule` (`layers.py:8`)
Pools per-node embeddings into a single graph descriptor. Soft-attention pooling:
```python
ctx = mean(H @ W)                  # (d,)  — global context
sigmoid(H @ tanh(ctx))             # (n,1) — attention weights per node
return H.t() @ weights             # (d,1) — weighted sum
```

### `TensorNetworkModule` (`layers.py:47`)
Bilinear similarity head between two pooled embeddings (originally from SimGNN paper):
```python
W ∈ R^(d, d, k=16) tensor + V ∈ R^(k, 2d) block + bias ∈ R^(k,1)
forward(p1, p2):
    a = p1.t() @ W @ p2                 # bilinear, (k,)
    b = V @ concat(p1, p2)              # block,    (k,)
    return ReLU(a + b + bias)           # (k,)
```
This is the "Neural Tensor Network" from the GedGNN paper — it produces the scalar bias term that gets added to the soft-mapping sum.

### `MatchingModule`, `GraphAggregationLayer`, `Mlp`, `sinkhorn`, `gumbel_softmax`
- `MatchingModule` is used by `GPN` (not GedGNN).
- `GraphAggregationLayer` is for `TaGSim` (also not us).
- `sinkhorn` is defined but **not actually used by GedGNN** — GedGNN uses plain row-wise softmax. (Worth knowing in case examiner asks "what about Sinkhorn for doubly-stochastic matrices?" — answer: it's available in this codebase but the authors found softmax sufficient.)

---

## 6. Post-processing — `src/kbest_matching_with_lb.py`

Inference-time only. Takes the soft mapping matrix from the model and produces a final discrete GED. Three classes:

### `GedLowerBound` (`:7`)
Computes a fast lower bound on the GED from a partial node mapping. Used to prune the search:
- Aligns sub-graphs implied by the partial mapping.
- Counts unmatched edges (`m1 - m2`) and unmatched node labels.
- Returns `(adjacency mismatch + label mismatch) / 2 + ...`.

### `Subspace` (`:65`)
Represents a subspace of the matching space defined by `(I, O)`: edges that must be used (I) and edges that must not (O). For each subspace, the best matching is computed (or given), and `get_second_matching` finds the second-best by O(n³) alternating-cycle search.

### `KBestMSolver` (further down the file)
Murty's algorithm: iteratively partitions the matching space, finds the best matching in each subspace, adds them to a heap, and pops the top-k. For each candidate matching, computes the actual GED of the resulting hard alignment, and returns the minimum.

**You don't need to know the algorithmic details.** You need to know:
- It's invoked at inference, not training.
- It explores `--postk` candidate matchings (default 1000).
- It computes actual GED for each candidate and returns the best.
- It's the source of the "k-best" / "Hungarian post-processing" terminology.

---

## 7. The training loop — `src/trainer.py` (the giant file, ~900 lines)

Most of it is administrative. The pieces that matter:

### `__init__(args)` (`trainer.py:21`)
Calls in order:
1. `setup_model()` — builds the chosen model (`GedGNN`, `SimGNN`, ...).
2. `load_data()` — reads JSONs.
3. `transfer_data_to_torch()` — pushes to GPU.
4. `init_graph_pairs()` — builds train/val/test pair lists.

### `fit()` (`trainer.py:430`)
One training epoch:
```python
random.shuffle(self.training_graphs)
for batch in chunks(training_graphs, batch_size=128):
    optimizer.zero_grad()
    losses = []
    for pair in batch:
        data = self.pack_graph_pair(pair)
        score, pre_ged, map_matrix = self.model(data)
        # value loss
        v_loss = MSE(score, data['target'])
        # mapping loss (only if gtmap supervision available)
        m_loss = fixed_mapping_loss(map_matrix, data['mapping'])
        losses.append(loss_weight * v_loss + m_loss)
    total = sum(losses) / len(losses)
    total.backward()
    optimizer.step()
```

(Approximate — the actual code is more verbose with logging.)

### `score(testing_graph_set)` (`trainer.py:482`)
Evaluation on val or test. Per pair:
1. Run model → get `score, pre_ged, map_matrix`.
2. Run `kbest_matching_with_lb` to refine prediction.
3. Compare to ground truth, accumulate metrics.

After the loop, aggregates:
- `mse, mae, acc, fea` — pointwise error metrics.
- `rho, tau` — rank correlations (computed per query graph, then averaged).
- `pk10, pk20` — precision@k computed by `cal_pk(num, pre, gt)`.

Then writes one TSV line to `result/results.txt`:
```
model	dataset	graph_set	#pairs	time/100p	mse	mae	acc	fea	rho	tau	pk10	pk20
GedGNN	AIDS	test	300	0.143	29.158	2.557	0.117	0.61	0.109	0.099	0.4	0.74
```

### `cal_pk(num, pre, gt)` (`trainer.py:467`) — the bug we fixed
Computes "what fraction of the top-`num` predicted-closest pairs are actually in the top-`num` true-closest pairs?" Original code crashed when fewer than `num` candidates existed; we patched it to clip `num` to the actual list length.

### `save(epoch)` / `load(epoch)` (`trainer.py:944, 949`)
Plain `torch.save(model.state_dict())` / `torch.load`. Path is `{model_path}{dataset}_{epoch}`.

---

## 8. Putting the run together — what happens when you call `main.py`

Trace, end to end, for `python src/main.py --model-name GedGNN --dataset AIDS --gnn-operator transformer --heads 4 --num-layers 3 --demo --model-epoch-end 1 --model-train 1`:

1. `param_parser` returns `args` with `gnn_operator='transformer'`, `heads=4`, etc.
2. `Trainer(args)`:
   - builds `GedGNN(args, num_labels=29)` → `setup_layers()` constructs the configurable transformer encoder (3 layers, 4 heads, no residual).
   - `load_data()` reads 700 AIDS graphs.
   - `transfer_data_to_torch()` builds edge_index / features tensors and the GT mapping/GED tables.
   - `init_graph_pairs()` sees `--demo`, restricts to 30/10/10, generates 465 training pairs from the first 30 graphs.
3. Outer loop (`epoch=0`):
   - `fit()` runs 1 inner epoch over 465 pairs, ~7 seconds on GPU.
   - `save(1)` writes the checkpoint.
   - `score('test')` evaluates on ~300 test pairs:
     - Model forward gives `(score, pre_ged, map_matrix)` per pair.
     - `kbest_matching_with_lb` refines using k=1000 candidates.
     - Aggregate metrics → write line to `result/results.txt`.
4. Done.

---

## 9. What you should be able to point at, file-by-file

| File | What's in it | Did we touch it? |
|------|---|---|
| `main.py` | 40-line entry point | No |
| `param_parser.py` | CLI flags | **Yes** (added 3 flags) |
| `models.py` | `SimGNN`, `GPN`, `GedGNN`, `TaGSim` classes | **Yes** (refactored GedGNN transformer branch only) |
| `GedMatrix.py` | `GedMatrixModule` (cost / map), various mapping losses | No |
| `layers.py` | `AttentionModule`, `TensorNetworkModule`, `sinkhorn`, etc. | No |
| `kbest_matching_with_lb.py` | `GedLowerBound`, `Subspace`, k-best Murty algorithm | No |
| `utils.py` | Data loading helpers | No |
| `trainer.py` | `Trainer` class — training, eval, demo handling, post-processing wiring | **Yes** (one bug fix in `cal_pk`) |

---

## 10. Three things to internalize

1. **The model's forward pass is short and beautiful.** Encoder → cost matrix + map matrix → soft sum + bias → score. Memorize this.
2. **The encoder is the only thing we touched.** Cost matrix, map matrix, NTN, k-best post-processing — all unchanged. So any change in metrics is causally attributable to the encoder.
3. **The matching matrix is bilinear.** It's not just a dot product `H1 @ H2.t()` — it's `H1 @ W @ H2.t()` with a learned 4D weight tensor and an MLP on top. This is why GedGNN can express richer pairwise interactions than vanilla similarity models.

After this walkthrough, go read [VIVA_PREP.md](VIVA_PREP.md). The viva prep assumes you understand the code; this doc gets you there.
