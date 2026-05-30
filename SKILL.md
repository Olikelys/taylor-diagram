---
name: taylor-diagram
description: "Create publication-quality Taylor diagrams in Python for multi-algorithm/multi-model metric comparison. Simultaneously displays standard deviation, correlation coefficient, and centered RMS difference on a single polar plot. Automatically handles dynamic standard deviation scaling and full negative correlation support. Triggers on: Taylor diagram, model comparison diagram, multi-model evaluation, algorithm metric comparison, skill metrics plot, 泰勒图, 泰勒图绘制, 模型性能对比图, 多算法指标对比."
metadata:
  version: 1.1.0
---

# Taylor Diagram Scientific Skill

## Overview

Create publication-quality Taylor diagrams that simultaneously display three key statistical metrics — standard deviation, correlation coefficient, and centered RMS difference — on a single polar coordinate plot. Featuring dynamic standard deviation scaling, full negative correlation support, and premium styling matching high-impact journals (e.g., Nature, Science).

## Workflow Decision Tree

```
User wants Taylor diagram?
├── Has observed + predicted arrays?
│   ├── Yes → Use TaylorDiagram class (Quick Start)
│   └── No → Need to compute from model results first?
│       └── Yes → See Integration with ML Workflow in references/examples.md
├── Prefers SkillMetrics library API?
│   └── Yes → Use plot_taylor_skillmetrics() or sm.taylor_diagram()
├── Need multi-panel (train/val/test)?
│   └── Yes → See Multi-Panel section in references/examples.md
├── Has negative correlations (range -1 to 1)?
│   └── Yes → Set corr_range=(-1, 1) to plot full semicircle
├── Data has raw, unnormalized standard deviations (e.g., standard deviation > 10)?
│   └── Yes → Set normalize=False (limits and ticks will auto-scale dynamically)
└── Many models (>5)?
    └── Yes → Use default Nature-inspired safe color cycle or colorbar mapping
```

## Quick Start

### Prerequisites

```bash
pip install numpy matplotlib
# Optional: for SkillMetrics API
pip install SkillMetrics
```

### Basic Usage

Import `TaylorDiagram` from `scripts/taylor_diagram`:

```python
import numpy as np
from taylor_diagram import TaylorDiagram

# observed: 1D reference values, pred_a/pred_b: 1D predictions
td = TaylorDiagram(observed, figsize=(8, 8))
td.add_model(pred_a, label="Model A", marker="o")
td.add_model(pred_b, label="Model B", marker="s")
td.plot(title="Taylor Diagram - Model Comparison")
td.savefig("taylor_diagram.png", dpi=300)
```

### Get Statistics Table

```python
print(td.get_stats_table())
# Model                     Std     Corr    CRMSD
# -----------------------------------------------
# Observed               1.0000   1.0000   0.0000
# Model A                1.0778   0.9559   0.3180
# Model B                0.9002   0.8091   0.5947
```

## Key API Reference

### TaylorDiagram Class

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observed` | array-like | (Required) | 1-D array of observed (reference) values |
| `fig` | plt.Figure | None | Existing figure to draw on. If None, created automatically |
| `rect` | int/tuple | 111 | Subplot specification |
| `figsize` | tuple | (8, 8) | Figure size in inches |
| `normalize` | bool | True | If True, normalize standard deviations by observed std (Observed std is at 1.0) |
| `corr_range` | tuple | (0, 1) | Correlation range. Set to `(-1, 1)` for negative correlations |
| `label_obs` | str | "Observed" | Label for observed reference point |
| `max_std` | float | None | Manual max standard deviation axis limit. If None, calculated dynamically |

| Method | Description |
|--------|-------------|
| `add_model(predicted, label, marker, color, markersize)` | Add a model prediction. If `color` is None, auto-selects from premium journal palette |
| `plot(show_crmsd, crmsd_levels, crmsd_color, ...)` | Render the diagram. Generates dynamic ticks and mathematically perfect CRMSD contours |
| `savefig(filename, dpi, bbox_inches)` | Save to file (PNG/PDF/SVG/TIFF) |
| `get_stats_table()` | Return formatted statistics table string |

## Premium Customizations & Edge Cases

### 1. Raw Ticks & Dynamic Scaling (e.g. Standard Deviations > 10)

Setting `normalize=False` enables plotting raw, unnormalized standard deviations. The axis limits and tick marks automatically adapt to cover the range of the observed and predicted standard deviations:

```python
td = TaylorDiagram(observed, normalize=False)
td.add_model(pred_a, label="SVR")
td.add_model(pred_b, label="Random Forest")
td.plot(title="Raw Taylor Diagram")
```

### 2. Full Correlation Range (-1.0 to 1.0)

When models are anti-correlated or correlations are negative, set `corr_range=(-1, 1)` to automatically extend the Taylor diagram to a full polar semi-circle:

```python
td = TaylorDiagram(observed, corr_range=(-1, 1))
td.add_model(pred_negative, label="Anti-Correlated Model")
td.plot(title="Semicircle Taylor Diagram")
```

### 3. Matplotlib 3.9+ Compatibility

The custom implementation avoids the deprecated `apply_theta_transforms` warnings by using standard polar transformations and explicitly handling layout, ensuring it is future-proof and robust across Matplotlib versions.

## Resources

### scripts/
- `taylor_diagram.py` — Core Taylor diagram implementation featuring the `TaylorDiagram` class, `compute_taylor_stats()`, and `plot_taylor_skillmetrics()`

### references/
- `taylor-theory.md` — Detailed theory, metric interpretation, and math definitions
- `skillmetrics-api.md` — SkillMetrics package API reference
- `examples.md` — 7 complete examples, updated for dynamic scaling, negative correlation semi-circles, and journal-ready layouts
