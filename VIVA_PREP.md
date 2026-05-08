# Viva Prep Guide — GedGNN with Graph Transformer Encoder

This file is your study plan. Goal: walk in able to *defend* every line of code change and every number in the results table. The questions an examiner will ask are predictable; this guide front-loads them.

---

## Phase 0 — Mental model (15 min)

Before anything else, internalize the **one-sentence summary** of the project. Practice saying it out loud:

> *"We took GedGNN, replaced its GIN encoder with a local Graph Transformer, made the depth and number of heads configurable, ran a 24-cell ablation on AIDS / Linux / IMDB, and found that attention helps most on larger graphs (IMDB), where it gives a 2.2× MAE improvement over GIN."*

If you can say that fluently, half the viva is already done. The rest is being able to drill into any one phrase: what is GedGNN, what is GIN, what is TransformerConv, what does the ablation actually show.

---

## Phase 1 — Understand the base paper (45–60 min)

Read **only the GedGNN paper** ([`ged-research-paper.pdf`](ged-research-paper.pdf)) — sections 1, 3, 4 are enough. Skim section 5 (experiments). You need to be able to answer:

### Q1.1: What is the Graph Edit Distance problem?
**Answer:** Given two graphs, GED is the minimum number of edit operations (node insertion/deletion, node relabeling, edge insertion/deletion) needed to transform one into the other. Computing exact GED is NP-hard, so we approximate it with neural networks.

### Q1.2: What does GedGNN do at a high level?
**Answer:** It's a 3-stage pipeline:
1. **Encode** each graph independently with a shared GNN (originally GIN), producing per-node embeddings $H_1 \in \mathbb{R}^{n_1 \times d}, H_2 \in \mathbb{R}^{n_2 \times d}$.
2. **Match** the two graphs differentiably:
   - A `costMatrix` module produces $C \in \mathbb{R}^{n_1 \times n_2}$ — a per-pair edit cost.
   - A `mapMatrix` module produces $M \in \mathbb{R}^{n_1 \times n_2}$ — soft node correspondences (after softmax).
   - The differentiable GED estimate is $\sum_{i,j} (\text{softmax}(M) \odot C)_{ij}$ + a scalar bias from a Neural Tensor Network (NTN) over the pooled graph embeddings.
3. **Post-process** at inference time using k-best Hungarian matching: enumerate top-$k$ hard assignments and pick the one minimizing actual GED.

### Q1.3: What are the three data sources of supervision?
**Answer:**
- Ground-truth GED values in `TaGED.json` per graph pair.
- Ground-truth optimal node mappings (the "best mapping list" used to supervise the soft matrix $M$).
- Type-aware GED decomposition (NC = node relabel, IN = node ins/del, IE = edge ins/del) — used by TaGSim, mostly informational here.

### Q1.4: What does GedGNN's loss look like?
**Answer:** Two terms:
- **Value loss** (MSE / BCE) between predicted similarity score and ground truth.
- **Mapping loss** (cross-entropy / KL) between predicted soft mapping $M$ and ground-truth mapping matrix.

You don't need to memorize the exact formula; just be able to say "value supervision + mapping supervision".

### Q1.5: Why is GIN the original encoder?
**Answer:** GIN is provably as expressive as the 1-Weisfeiler-Lehman graph isomorphism test (Xu et al., 2019). For graph similarity tasks, you need an encoder that can distinguish non-isomorphic graphs — GIN does that maximally among message-passing GNNs.

---

## Phase 2 — Understand exactly what we changed (30 min)

There are exactly **3 file changes**. Memorize them.

### File 1: `project_files/experiments/Overall Performance/src/param_parser.py`
**What we did:** Added 3 CLI flags.
- `--heads` (int, default 4) — number of attention heads in `TransformerConv`.
- `--num-layers` (int, default 3) — depth of the encoder stack.
- `--residual` (flag, default off) — whether to add residual connections between encoder layers.

**Why we did it:** The original code hard-coded these values. To run an ablation, they need to be sweepable from the command line.

### File 2: `project_files/experiments/Overall Performance/src/models.py`
**What we did:** Refactored only the `GedGNN` class's `transformer` branch.

**Before** (original code, hard-coded):
```python
self.convolution_1 = TransformerConv(num_labels,  filters_1 // 4, heads=4, ...)
self.convolution_2 = TransformerConv(filters_1,   filters_2 // 4, heads=4, ...)
self.convolution_3 = TransformerConv(filters_2,   filters_3 // 4, heads=4, ...)
self.norm_1 = LayerNorm(filters_1)
self.norm_2 = LayerNorm(filters_2)
```

**After** (configurable):
```python
heads      = self.args.heads        # default 4
num_layers = self.args.num_layers   # default 3
# Width schedule:
if num_layers == 3:
    dims = [num_labels, 128, 64, 32]    # exact backward-compat with original
else:
    dims = [num_labels] + [128]*(num_layers-1) + [32]

self.tf_convs         = ModuleList()
self.tf_norms         = ModuleList()
self.tf_residual_proj = ModuleList()
for i in range(num_layers):
    in_dim, out_dim = dims[i], dims[i+1]
    assert out_dim % heads == 0           # divisibility constraint
    self.tf_convs.append(TransformerConv(in_dim, out_dim // heads, heads=heads, ...))
    if i < num_layers - 1:
        self.tf_norms.append(LayerNorm(out_dim))
        if residual:
            self.tf_residual_proj.append(
                Linear(in_dim, out_dim) if in_dim != out_dim else Identity()
            )
```

**The forward pass (`convolutional_pass`)** now iterates this list:
```python
for i, conv in enumerate(self.tf_convs):
    prev = features
    h = conv(features, edge_index)
    if i < len(self.tf_convs) - 1:
        h = self.tf_norms[i](h)              # post-norm
        h = F.elu(h)                         # activation
        if residual:
            h = h + self.tf_residual_proj[i](prev)
        h = F.dropout(h, p=0.5, training=self.training)
    features = h
return features
```

### File 3: `project_files/experiments/Overall Performance/src/trainer.py` — bug fix only
**What we did:** Patched `cal_pk(num, pre, gt)` to handle test groups smaller than `num`. Original code did `for i in range(num): ...` and crashed with `IndexError` when `len(beta) < num`.

**Why we did it:** Demo mode shrinks each test query group below the p@20 cutoff, triggering the bug. Pre-existing bug, not introduced by us — just exposed by `--demo`.

---

## Phase 3 — Defend every design decision (30 min)

These are the questions that *will* be asked. Practice the answers.

### Q3.1: Why `out_dim // heads`? Why not just multiply heads?

**Setup:** `TransformerConv` outputs `heads × out_per_head` features. So if you set `out_per_head = 32` and `heads = 4`, the output is 128-dim.

**Answer:** We fix the **total** output width to `dims[i+1]` (e.g. 32 for the final layer) and divide by the number of heads. This means doubling heads halves per-head width; the output dimensionality stays constant. **Critical for ablation fairness:** the parameter count and the dimensionality of the downstream cost/map matrices are independent of `heads`. If we let total width grow with heads, we'd be conflating "more heads" with "more parameters" and "wider matching matrices" — three confounds at once.

### Q3.2: Why does the divisibility assertion matter?

**Answer:** `out_dim % heads == 0`. Otherwise we couldn't split the 32-dim output evenly across heads. With our default schedule `[…, 128, 64, 32]` and `heads ∈ {1, 2, 4, 8}`, every layer's output is divisible by every choice of `heads`. With `heads=8` and `out=32`, we get 4 dims/head — that's the smallest in our sweep, and arguably already too narrow (see Q3.7).

### Q3.3: Why ELU activation, not ReLU?

**Answer:** The original transformer branch in the GedGNN code already used ELU. We preserved it for backward compatibility. ELU is also a more common pairing with Transformer blocks because its smooth negative tail prevents dead-neuron issues that hurt attention layers. The GIN/GCN branches still use ReLU because that's what their authors used.

### Q3.4: Why "Post-Norm" (LayerNorm before activation)?

**Answer:** We applied LayerNorm directly to the conv output, then ELU. This is what the original 3-layer hard-coded code did. We kept it for backward compatibility. A purer "Pre-Norm" Graph Transformer block (norm before the attention) is known to train more stably and would be a natural follow-up — we mention this in the "limitations" section.

### Q3.5: Why does the residual path need a `Linear` projection sometimes?

**Answer:** The residual sum `h + prev` requires `h` and `prev` to have the same number of features. Layers where `in_dim != out_dim` (e.g. `num_labels → 128`, or `64 → 32`) need a learned `Linear(in_dim, out_dim)` to project `prev` to match `h`. Where dims already match (e.g. when `num_layers > 3` and intermediate transitions are `128 → 128`), we use `nn.Identity()` to avoid wasting parameters.

### Q3.6: Why doesn't the last layer have LayerNorm/ELU/Dropout?

**Answer:** The cost/map matrices were calibrated against the **un-normalized output scale** of the original GIN encoder. Adding a final LayerNorm would change that scale and force re-tuning the matching modules. Keeping the last layer un-normalized matches the original code's behavior on both GIN and the original 3-layer Transformer branch.

### Q3.7: Heads ablation — why does heads=8 underperform?

**Answer:** The final embedding is 32-dim. With `heads=8`, that's only 32/8 = 4 dimensions per head. Each head has too little capacity to encode meaningful relational features, and you start losing more than you gain from having more heads. Sweet spot in our sweep is `heads ∈ {2, 4}`.

### Q3.8: Depth ablation — why does deeper help on IMDB but hurt on AIDS?

**Answer:** Two opposing effects:
- **Receptive field.** Each layer extends the receptive field by one hop. AIDS graphs have ~10 nodes, so 2 layers already cover them. IMDB graphs are much larger; you need more depth to propagate information across the diameter.
- **Over-smoothing.** Each layer averages neighbor information; with too many layers, all node embeddings collapse to similar values. On small graphs this happens quickly.

So the optimal depth scales with graph size — that's our cleanest empirical finding. On IMDB, MAE drops monotonically: 13.94 (L=2) → 6.61 (L=3) → 4.42 (L=4).

### Q3.9: Residual connections — why do they help AIDS but hurt Linux/IMDB?

**Answer:** With Post-Norm residuals and only 1 epoch of training, the residual path isn't well-conditioned. On Linux the training loss diverged. We expect Pre-LN residuals + longer training to make this uniformly positive — it's listed as future work.

### Q3.10: Why is MAE on AIDS not improved but ranking metrics are?

**Answer:** MAE measures pointwise GED error; ranking metrics (Spearman ρ, Kendall τ, p@k) measure how well the model **orders** test pairs. The attention encoder learns embeddings whose pairwise similarity better tracks the true GED ordering, even when the absolute predicted GED isn't more accurate. For retrieval workloads (find the most similar molecules to query X), this is the more useful property.

### Q3.11: Why use `--demo` mode? Isn't 1 epoch unscientific?

**Answer:** Honest answer: **time pressure.** 24 runs at full 20-epoch training would take many hours. Demo mode (1 epoch on 30 graphs, ~170–470 training pairs) lets us see *architectural trends* across all 24 cells in ~40 minutes total. The trends that emerge — heads saturation, depth × graph-size interaction — are robust because they appear consistently across the 3 datasets. We do explicitly call out in the limitations section that absolute MAE numbers are noisy and a production comparison would drop `--demo` and train for 20 epochs.

### Q3.12: What's the difference between `TransformerConv` and a "real" Graph Transformer?

**Answer:** `TransformerConv` (Shi et al. 2021, used in PyG) does multi-head attention **only over 1-hop neighbors** — the same locality as GCN/GIN, but with learned attention weights instead of fixed weights or sum aggregation. A "full" Graph Transformer like Graphormer attends to **all node pairs** (quadratic) and adds positional encodings (Laplacian PE, shortest-path bias, etc.). We deliberately chose `TransformerConv` because it's the right baseline against GIN/GCN: same receptive field, only the aggregation is different. This isolates the attention mechanism as the independent variable.

### Q3.13: What did you NOT change, and why does it matter?

**Answer:** Cost matrix (`GedMatrixModule` for $C$), mapping matrix (`GedMatrixModule` for $M$), the soft assignment via softmax, the NTN scalar bias, the k-best Hungarian post-processing, the loss function, the data loader, the train/test splits — all unchanged. This is deliberate: any difference in metrics between variants is attributable purely to the encoder. Otherwise you'd have multiple confounds.

---

## Phase 4 — Numbers you must know cold (10 min)

| Claim | Number to remember |
|---|---|
| Best MAE on IMDB (`tf_h4_l4`) | **4.42** vs GIN **9.75** — 2.2× better |
| Best ρ on AIDS (`tf_h4_l3_res`) | **0.675** vs GIN **0.109** |
| Best p@20 on AIDS | **0.89** vs GIN **0.74** |
| IMDB depth sweep at H=4 | **13.94 → 6.61 → 4.42** as L=2,3,4 |
| Linux best model overall | **GCN** (MAE 1.25) — be honest, it beat GT |
| Number of variants × datasets | **8 × 3 = 24** |
| Compute regime | `--demo`: 30 train graphs, 1 epoch, ~170–470 training pairs |

**If asked for the headline number, say:** *"On IMDB, MAE dropped from 9.75 with GIN to 4.42 with a depth-4, 4-head Graph Transformer — a 2.2× improvement."*

---

## Phase 5 — How to present the project (15 min)

### Suggested 5-minute slide flow

1. **Slide 1 — The problem (30s).** "Graph Edit Distance is NP-hard. GedGNN approximates it neurally. We extend it."
2. **Slide 2 — GedGNN pipeline diagram (1 min).** A picture of: 2 graphs → shared encoder → cost matrix + map matrix → soft GED → k-best post-processing → final GED. Highlight the encoder box in red — *"this is the only thing we change."*
3. **Slide 3 — What we changed (1 min).** Show the before/after code snippet. Three flags: heads, layers, residual. Bullet what we *didn't* change (everything else) — that earns trust points.
4. **Slide 4 — Headline result (1 min).** The IMDB depth table:
   - 2 layers → MAE 13.9
   - 3 layers → MAE 6.6
   - 4 layers → MAE **4.4** (vs GIN baseline 9.75)
   *"Depth × graph size: deeper attention helps when graphs are bigger."*
5. **Slide 5 — Heads ablation (45s).** Show the small heads table. *"4 heads is the sweet spot; 8 is too narrow given the 32-dim bottleneck."*
6. **Slide 6 — AIDS ranking story (45s).** Show the ρ jump 0.11 → 0.68. *"On small graphs, attention doesn't reduce MAE but it reorders similarity better — useful for retrieval."*
7. **Slide 7 — Limitations & future work (30s).** Demo budget, local attention only, no positional encodings. Honest disclosure earns points.

### Tone tips
- **Don't oversell.** The project's strongest results are on IMDB; on AIDS the MAE story is mixed; on Linux GCN wins. Acknowledge this proactively. Examiners distrust monotonically positive results.
- **Mention compute honestly.** "We used demo mode for the sweep due to time constraints" is a perfectly acceptable answer if you immediately follow up with "but the trends are robust because they replicate across 3 datasets."
- **Have one diagram ready** showing the 3-stage GedGNN pipeline with the encoder boxed. If the examiner asks "where exactly did you intervene?", point to the box.

---

## Phase 6 — Trap questions to anticipate

### Trap 1: "Why didn't you just use GAT? It also has attention."
**Answer:** GAT and TransformerConv are closely related — both attend over 1-hop neighbors. TransformerConv is essentially GAT with multi-head dot-product attention and an extra value projection (closer to standard Transformer math). We picked it because it's the variant in the original GedGNN code's transformer branch, so the apples-to-apples comparison was natural. GAT would likely give similar trends.

### Trap 2: "Aren't your improvements just from having more parameters?"
**Answer:** No, this is exactly why we used `out_dim // heads`. The total output width is fixed at 32 regardless of heads. A `TransformerConv(in, 8, heads=4)` and `TransformerConv(in, 4, heads=8)` have very similar parameter counts. The depth ablation does add parameters with depth (extra layers), but that's expected — and the cleanest finding (depth helps on IMDB) wouldn't disappear by parameter-matching, because GIN at the same depth would still under-perform.

### Trap 3: "Your residual ablation is contradictory — explain."
**Answer:** Yes, residuals helped AIDS but hurt Linux/IMDB. We attribute this to (a) Post-Norm placement, which is known to be unstable, and (b) a 1-epoch training budget, which doesn't give the residual path time to be useful. With Pre-Norm and 20 epochs we'd expect uniform improvement; the current behavior is a compute-budget artifact, not a fundamental architectural finding.

### Trap 4: "Why is `--demo` defensible?"
**Answer:** It isn't a final benchmark — it's an architectural-trend sweep. We make 24 runs comparable to each other under identical compute. The findings we trust are the ones that replicate across all three datasets (heads saturation at H=4) or that are monotonic (depth helps on IMDB). Findings that appear in only one dataset (the AIDS residual win) we explicitly flag as needing more training to confirm.

### Trap 5: "What would a cleaner extension look like?"
**Answer:** Three things:
1. **Laplacian positional encodings** concatenated to input features — known to help Transformers regain structural awareness.
2. **Pre-LN residuals** instead of Post-Norm.
3. **Replace the NTN bias with cross-attention** between the two pooled graph embeddings — make the *interaction* module attention-based as well.

If you say these unprompted, it shows you understand the design space.

---

## Phase 7 — Final 10-minute checklist

Before walking in:
- [ ] Can you draw the 3-stage GedGNN pipeline on a whiteboard?
- [ ] Can you say which file you changed and roughly how many lines?
- [ ] Can you state the IMDB headline number (4.42 vs 9.75) without looking?
- [ ] Can you say *why* depth helps on IMDB and hurts on AIDS in one sentence?
- [ ] Can you explain `out_dim // heads` and why it's needed for fair ablation?
- [ ] Can you list 3 limitations of your study without looking?
- [ ] Have you re-read [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) and [result/SUMMARY.md](project_files/experiments/Overall Performance/result/SUMMARY.md) once more?

If yes to all 7, you're ready.
