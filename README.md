# Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**k = 3 is the critical threshold for representing cyclic processes without temporal conflation.**

Submitted to **Information Geometry** (Springer) — INGE-D-25-00099, under review.

## Abstract

We establish that k=3 is the minimal embedding dimension for self-intersection-free representation of cyclic processes with monotone meta-time on statistical manifolds. For *phase-preserving* embeddings (where the image depends only on phase φ, not meta-time τ), k ≤ 2 forces categorical representations through unavoidable state conflation, while k ≥ 3 preserves continuous temporal dynamics.

## Key Results

1. **Non-existence in ℝ²**: Any phase-preserving map π₂: K → ℝ² (i.e., π₂|_K = f ∘ φ|_K) necessarily has self-intersections.

2. **Existence in ℝ³**: The canonical helix embedding π₃(γ(t)) = (cos(2πφ), sin(2πφ), τ/T) is injective.

3. **Discretization as Quotient**: For k ≤ 2, the equivalence classes form a quotient homeomorphic to S¹—the meta-time information is "quotiented out."

## Companion Paper

This paper forms a two-paper program with:

**[Quotient Geometry of Statistical Manifolds Under Dimensional Collapse](https://github.com/todd866/code-emergence)** (in preparation)

- **This paper**: establishes the specific k = 3 threshold for cyclic processes with monotone meta-time
- **Companion paper**: provides the general quotient-geometric framework (fiber structure, metric descent, covering numbers) that makes this phenomenon inevitable

The papers share notation: φ for phase coordinate, τ for meta-time, V = ker(dφ) for vertical distribution (meta-time directions), H = V^⊥ for horizontal.

## Repository Structure

```
├── dimensional_collapse.tex     # Submitted manuscript
├── dimensional_collapse.pdf     # Compiled PDF
├── generate_figures.py          # Figure generation script
├── figures/                     # Generated figures
├── references.bib               # Bibliography
└── revisions/
    └── dimensional_collapse_r1.tex  # R1 revision (in progress)
```

### Revisions

The `revisions/` directory contains refinements addressing cross-paper consistency:

- **Phase coordinate renamed**: θ → φ (avoiding conflict with parameter θ in p_θ)
- **Domain clarified**: π_k defined on trajectory K = γ([0,T]), not all of M
- **Phase-preserving simplified**: Now "factors through phase" (π₂|_K = f ∘ φ|_K)
- **Fisher metric clarified**: Coordinate-free statement of rank drop; connects to companion paper's fiber structure
- **Bridge remark added**: Explicit connection to quotient geometry (V, H, fiber foliation)

## Figures

```bash
python generate_figures.py
```

Generates:
- `fig1_collision_problem.pdf` — Self-intersection visualization (helix vs. projection)
- `fig2_fisher_rank.pdf` — Information geometry of rank drop
- `fig3_general_cycles.pdf` — Generalization to directed cycles

## Requirements

```
numpy
matplotlib
scipy
```

## Citation

```bibtex
@article{todd2025minimal,
  title={Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes on Statistical Manifolds},
  author={Todd, Ian},
  journal={Information Geometry},
  year={2025},
  note={Under review, INGE-D-25-00099}
}
```

## License

MIT License
