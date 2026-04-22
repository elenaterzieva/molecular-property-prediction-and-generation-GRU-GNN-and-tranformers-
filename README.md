# Molecular Property Prediction DL
*Comparing geometric (3D) vs. textual (SMILES) representations for molecular formation energy prediction.*

---

## Overview

This project benchmarks two fundamentally different ways to represent molecules for a regression task: predicting **formation energy** (kJ/mol) of ~129 000 small organic molecules from a QM9-style dataset.

| Representation | Model | Val MAE | Val RMSE | R² |
|---|---|---|---|---|
| SMILES (text) | Bidirectional GRU | 0.34 kJ/mol | 0.53 kJ/mol | ~0.95 |
| Geometry (3D coords) | PaiNN (equivariant GNN) | **0.11 kJ/mol** | **<0.15 kJ/mol** | **0.9998** |

---

## Tasks

### Task 1 — Formation Energy Prediction
Two models built from scratch and trained independently:

- **SmilesGRU** — Tokenised SMILES strings → Embedding → Bidirectional GRU (2 layers, hidden=256) → mean-attention pooling → MLP regression head. SMILES augmentation used during training to avoid overfitting to canonical syntax.
- **GeometryNet / PaiNN** — 3D atomic coordinates + atom types → radius graph (cutoff 5 Å) → 3 PaiNN interaction layers (Bessel RBF, cosine cutoff) → global mean pooling → MLP regression head. Respects rotational equivariance by design.

### Task 2 — Data Efficiency
Both models retrained on subsets of increasing size (100 → 300 → 1 000 → 3 000 → 10 000 molecules) to compare sample efficiency across representations.

### Task 3 — SMILES Generation
A Transformer-based generative model trained to produce novel valid SMILES strings, evaluated on **validity**, **uniqueness**, and **novelty**.

---

## Dataset

- **129 012 molecules** (train: 119 012 / test: 10 000)
- Atoms: H, C, N, O, F, S
- Labels: normalised formation energy (μ, σ stored for denormalisation)
- Files in `ass2_data/`: `formation_energy.npz`, `pos_data.pkl`, `type_data.pkl`, `smiles.pkl`, `data_split.npz`

---

## Key Files

```
ass2.ipynb                  # Main solution notebook (all three tasks)
geom_sources/PaiNN.py       # PaiNN equivariant message-passing model
smiles_model_sources/       # Transformer + tokeniser + augmenter utilities
best_geometry.pt            # Saved PaiNN checkpoint
best_smiles.pt              # Saved GRU checkpoint
task1_smiles_predictions.npy  # Test-set predictions (SMILES model)
```

---

## Training Details

| Setting | Value |
|---|---|
| Optimiser | Adam |
| LR schedule | Cosine annealing |
| Early stopping | Patience = 5 epochs |
| Gradient clipping | Yes |
| GRU dropout | 0.1 |
| PaiNN RBF features | 20 Bessel basis functions |

---

## Main Finding

3D geometry with built-in equivariance (PaiNN) substantially outperforms sequence-based SMILES (GRU) for formation energy prediction, confirming that spatial structure is the dominant information source for this quantum-chemical property.
