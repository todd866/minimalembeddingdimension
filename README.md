# Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes

Code and figures for the paper:

**Minimal Embedding Dimension for Self-Intersection-Free Recurrent Processes on Statistical Manifolds**

Ian Todd, University of Sydney

Submitted to *Information Geometry* (Springer)

## Abstract

We establish that k=3 is the minimal embedding dimension for self-intersection-free representation of cyclic processes with monotone meta-time on statistical manifolds. This identifies a critical threshold in information geometry: k≤2 forces categorical representations through unavoidable state conflation, while k≥3 preserves continuous temporal dynamics.

## Contents

- `generate_figures.py` - Python script to reproduce all figures
- `dimensional_collapse.tex` - LaTeX source
- `references.bib` - Bibliography
- `figures/` - Generated figure PDFs

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
