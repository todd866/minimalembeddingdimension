# Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes

Code and figures for the paper:

**Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes on Statistical Manifolds**

Ian Todd, University of Sydney

Submitted to *Information Geometry* (Springer)

## Abstract

We establish that k=3 is the minimal embedding dimension for self-intersection-free representation of cyclic processes with monotone meta-time on statistical manifolds. This identifies a critical threshold in information geometry: k≤2 forces categorical representations through unavoidable state conflation, while k≥3 preserves continuous temporal dynamics.

## Contents

- `generate_figures.py` - Python script to reproduce all figures
- `dimensional_collapse.tex` - LaTeX source (submitted version)
- `references.bib` - Bibliography
- `figures/` - Generated figure PDFs

### Revisions folder

The `revisions/` directory contains post-submission refinements addressing potential reviewer concerns:

- **Topological clarification:** The original theorem statement could be read as claiming the cylinder $S^1 \times \mathbb{R}$ cannot embed in $\mathbb{R}^2$ (it can, as a spiral). The revision explicitly restricts to *phase-preserving* embeddings—those where the image depends only on phase, not meta-time—which is the physically meaningful constraint for limit-cycle systems.
- **Tightened proofs:** Minimality claim now explicitly scoped to "within the phase-preserving class."
- **Fisher metric clarification:** Distinguished intrinsic metric behavior from projection-induced rank loss.

The root-level files reproduce the submitted PDF exactly. The `revisions/` folder contains work-in-progress improvements for potential revision requests.

## Requirements

```
numpy
matplotlib
scipy
```

## Usage

```bash
python generate_figures.py
```

This generates:
- `figures/fig1_collision_problem.pdf` - Self-intersection visualization
- `figures/fig2_fisher_rank.pdf` - Information geometry analysis
- `figures/fig3_general_cycles.pdf` - Generalization to directed cycles

## License

MIT
