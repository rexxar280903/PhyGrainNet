# PhyGrainNet Architecture

## Design principles

The primary model is custom, compact, and trained from scratch. The architecture injects domain properties that generic classifiers do not explicitly enforce: physical scale, granular texture, multiple views of one material sample, and cumulative-distribution validity.

## PhyGrainNet v0

```text
image
  ↓
stem convolution
  ↓
GrainBlock × N
  ↓
global average pooling
  ↓
MLP embedding
  ↓
11 logits
  ↓
softmax × 100
  ↓
11 non-negative interval masses
  ↓
cumulative sum
  ↓
valid GSD
```

## GrainBlock

```text
input
 ├─ depthwise 3×3 ──────────┐
 └─ dilated depthwise 3×3 ──┤
                            concat
                              ↓
                           1×1 conv
                              ↓
                           GroupNorm
                              ↓
                             GELU
                              ↓
                         channel gate
                              ↓
                           residual
```

GroupNorm is used because the independent sample count may require small batches.

## Constrained distribution head

For logits (z_i), interval masses are

```math
m_i = 100\frac{e^{z_i}}{\sum_j e^{z_j}}
```

and cumulative outputs are

```math
\hat F_k = \sum_{i=1}^{k} m_i.
```

Therefore:

```math
0 \le \hat F_1 \le \cdots \le \hat F_{11}=100.
```

## Candidate full architecture

```text
multiple sample views
        ↓
PPM-aware physical crop
        ↓
candidate physical scales (10 / 20 / 40 mm)
        ↓
  RGB stream + texture stream
        ↓
cross-scale fusion
        ↓
per-view embedding
        ↓
mean + std + attention aggregation
        ↓
sample embedding
        ├── auxiliary composition head
        └── constrained GSD head
```

The physical scales are provisional until the Kaggle dataset audit confirms real PPM values and usable image dimensions.

## Acceptance rule

A module stays in the final architecture only when a controlled ablation improves grouped validation or enforces a necessary physical invariant.
