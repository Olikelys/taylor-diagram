# Skill Name: taylor-diagram

[TOC]

## Overview

Create publication-quality Taylor diagrams in Python for multi-algorithm/multi-model metric comparison. Simultaneously displays standard deviation, correlation coefficient, and centered RMS difference on a single polar coordinate plot.

## Features

* Self-contained `TaylorDiagram` class — no external dependencies beyond numpy/matplotlib
* Simultaneous visualization of Std, Correlation, and CRMSD on one diagram
* Support for normalized and non-normalized standard deviations
* Configurable correlation range (0,1) or (-1,1)
* Automatic CRMSD contour arcs from reference point
* Formatted statistics table output
* SkillMetrics library wrapper for users who prefer that API
* Multi-panel diagram support for train/val/test comparisons
* Colorbar mapping mode for many-model comparisons (>5 models)
* Publication-quality export (PNG, PDF, SVG)

## Prerequisites & Setup

```bash
pip install numpy matplotlib
# Optional: for SkillMetrics API
pip install SkillMetrics
```

## Quick Start

```python
import numpy as np
from taylor_diagram import TaylorDiagram

np.random.seed(42)
observed = np.random.randn(200) + 5
pred_a = observed + np.random.randn(200) * 0.3
pred_b = observed * 0.8 + np.random.randn(200) * 0.5 + 1

td = TaylorDiagram(observed, figsize=(8, 8))
td.add_model(pred_a, label="Model A", marker="o", color="tab:blue")
td.add_model(pred_b, label="Model B", marker="s", color="tab:orange")
td.plot(title="Taylor Diagram - Model Comparison")
td.savefig("taylor_diagram.png", dpi=300)
```

## Trigger Prompts & User Scenarios

- **Scenario 1: Multi-Model Comparison**
  - *User Prompt:* "画一个泰勒图对比我的5个回归模型"
  - *Expected Behavior:* Agent uses TaylorDiagram class to create diagram with all models

- **Scenario 2: Publication Figure**
  - *User Prompt:* "Create a Taylor diagram for my paper comparing SVR, RF, XGBoost predictions"
  - *Expected Behavior:* Agent creates publication-quality diagram with proper styling

- **Scenario 3: Statistics Only**
  - *User Prompt:* "计算各模型的标准差、相关系数和中心化均方根误差"
  - *Expected Behavior:* Agent uses compute_taylor_stats() and outputs table

- **Scenario 4: SkillMetrics API**
  - *User Prompt:* "用SkillMetrics库画泰勒图"
  - *Expected Behavior:* Agent uses plot_taylor_skillmetrics() wrapper

## Input & Output Specification

**Inputs**
- `observed`: 1-D array of reference/observed values
- `predicted`: 1-D array(s) of model prediction values
- Optional: styling parameters (markers, colors, labels, etc.)

**Outputs**
- Matplotlib figure with Taylor diagram
- Optional: PNG/PDF/SVG file
- Optional: formatted statistics table string

## Limitations & Known Issues

- **Classification only**: Taylor diagrams are for continuous predictions, not classification tasks
- **CRMSD vs RMSE**: CRMSD removes mean bias; models with large bias may appear better than expected
- **Correlation clipping**: Correlation values are clipped to [-1, 1] for arccos computation
- **Matplotlib version**: Some deprecation warnings on matplotlib >= 3.9 (cosmetic only)

## Acknowledgments & References

- Taylor, K. E. (2001). Summarizing multiple aspects of model performance in a single diagram. *J. Geophys. Res.*, 106(D7), 7183–7192.
- SkillMetrics: https://github.com/PeterRochford/SkillMetrics

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE.txt) file for details.
