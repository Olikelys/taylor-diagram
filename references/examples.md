# Taylor Diagram Examples

This guide contains complete, ready-to-run examples demonstrating the enhanced features of the `TaylorDiagram` class.

## Table of Contents

1. [Basic Usage - Standard Normalized Diagram](#1-basic-usage---standard-normalized-diagram)
2. [Dynamic Scaling with Raw Data (Standard Deviation > 100)](#2-dynamic-scaling-with-raw-data-standard-deviation--100)
3. [Negative Correlations & Semi-Circle Plot](#3-negative-correlations--semi-circle-plot)
4. [Publication-Quality Aesthetics (Nature Style)](#4-publication-quality-aesthetics-nature-style)
5. [Multi-Panel Subplots layout](#5-multi-panel-subplots-layout)
6. [Statistics Table and Math Verification](#6-statistics-table-and-math-verification)
7. [Integration with Scikit-Learn ML Pipeline](#7-integration-with-scikit-learn-ml-pipeline)

---

## 1. Basic Usage - Standard Normalized Diagram

A clean, standard normalized Taylor diagram (Observation standard deviation is normalized to 1.0):

```python
import numpy as np
from taylor_diagram import TaylorDiagram

np.random.seed(42)
observed = np.random.randn(200) + 5
pred_a = observed + np.random.randn(200) * 0.3
pred_b = observed * 0.85 + np.random.randn(200) * 0.5 + 0.7

td = TaylorDiagram(observed, figsize=(8, 8))
td.add_model(pred_a, label="Algorithm A", marker="o")
td.add_model(pred_b, label="Algorithm B", marker="s")
td.plot(title="Normalized Taylor Diagram Comparison")
td.savefig("taylor_basic.png", dpi=300)
```

## 2. Dynamic Scaling with Raw Data (Standard Deviation > 100)

When standard deviations of data are arbitrary (not normalized), set `normalize=False`. The standard deviation axis and ticks will scale automatically to fit your data perfectly:

```python
import numpy as np
from taylor_diagram import TaylorDiagram

np.random.seed(42)
# Observations with standard deviation of around 120
observed = np.random.randn(500) * 120 + 500

# Models with raw values
models = {
    "Model Alpha": observed + np.random.randn(500) * 35,
    "Model Beta": observed * 0.85 + np.random.randn(500) * 60 + 50,
    "Model Gamma": observed * 1.2 + np.random.randn(500) * 80 - 100
}

# Set normalize=False for raw values
td = TaylorDiagram(observed, normalize=False, figsize=(9, 9))

for name, pred in models.items():
    # Colors will be automatically assigned from the premium Nature palette
    td.add_model(pred, label=name, marker="o", markersize=10)

td.plot(title="Taylor Diagram - Raw Values Comparison")
td.savefig("taylor_raw_dynamic.png", dpi=300)

# Print a highly-readable statistics table
print(td.get_stats_table())
```

## 3. Negative Correlations & Semi-Circle Plot

If some of your algorithms or models are anti-correlated with observations, set `corr_range=(-1, 1)`. The polar sector will automatically expand to a perfect semicircle ($0$ to $180^\circ$):

```python
import numpy as np
from taylor_diagram import TaylorDiagram

np.random.seed(42)
observed = np.random.randn(300)

# Standard predictions (positive correlation)
pred_pos = observed * 0.9 + np.random.randn(300) * 0.3
# Anti-correlated predictions (negative correlation)
pred_neg = -observed * 0.8 + np.random.randn(300) * 0.4

td = TaylorDiagram(observed, corr_range=(-1, 1), figsize=(10, 8))
td.add_model(pred_pos, label="Positive Corr Model", marker="o")
td.add_model(pred_neg, label="Anti-Correlated Model", marker="D")

# Ticks and mathematically perfect CRMSD contours will cover the entire range
td.plot(title="Semicircle Taylor Diagram (-1.0 to 1.0)")
td.savefig("taylor_semicircle.png", dpi=300)
```

## 4. Publication-Quality Aesthetics (Nature Style)

Follows `nature-figure` aesthetics to generate publication-grade illustrations:

```python
import numpy as np
from taylor_diagram import TaylorDiagram

np.random.seed(123)
obs = np.random.randn(400) * 10 + 20

models = {
    "SVR": obs * 0.95 + np.random.randn(400) * 2.0,
    "RF": obs * 0.90 + np.random.randn(400) * 3.5,
    "XGBoost": obs * 0.98 + np.random.randn(400) * 1.5,
    "LSTM": obs * 0.82 + np.random.randn(400) * 5.0,
}

# Journal dimensions: single column standard width (89 mm / 3.5 inches)
# Use a refined figsize and font size
td = TaylorDiagram(obs, figsize=(6.5, 6.5), normalize=True)

markers = ["o", "s", "^", "D"]
for (name, pred), marker in zip(models.items(), markers):
    # Set explicit markers, auto-color from high-impact palette
    td.add_model(pred, label=name, marker=marker, markersize=8)

# Customize CRMSD contour line styles to be delicate and elegant
td.plot(
    title="Algorithm Performance Comparison",
    crmsd_color="#c0c0c0",
    crmsd_style=":",
    crmsd_alpha=0.6,
    obs_marker="*",
    obs_color="#ff3333",  # Highlight the observation reference point in red
    obs_markersize=14
)

# Export standard publication formats (vector and high DPI)
td.savefig("taylor_publication.svg", bbox_inches="tight")
td.savefig("taylor_publication.pdf", bbox_inches="tight")
td.savefig("taylor_publication.png", dpi=600, bbox_inches="tight")
```

## 5. Multi-Panel Subplots Layout

Generate side-by-side Taylor diagrams to compare performance across different splits (Train / Val / Test) or different scenarios:

```python
import numpy as np
import matplotlib.pyplot as plt
from taylor_diagram import TaylorDiagram

np.random.seed(42)
fig = plt.figure(figsize=(15, 5))

splits = ["Training Split", "Validation Split", "Testing Split"]
obs_splits = [np.random.randn(200) + 10 for _ in range(3)]

for i, (split_name, obs) in enumerate(zip(splits, obs_splits)):
    pred_a = obs * 0.95 + np.random.randn(200) * (0.2 + i * 0.1)
    pred_b = obs * 0.85 + np.random.randn(200) * 0.5
    
    # Grid layout: 1 row, 3 columns (Subplots 131, 132, 133)
    td = TaylorDiagram(obs, fig=fig, rect=131 + i, figsize=(15, 5))
    td.add_model(pred_a, label="Model Alpha", marker="o")
    td.add_model(pred_b, label="Model Beta", marker="s")
    
    # Legend is only plotted on the first panel to avoid cluttering
    td.plot(title=f"({chr(97+i)}) {split_name}", legend=(i == 0))

plt.tight_layout()
plt.savefig("taylor_subplots.png", dpi=300, bbox_inches="tight")
```

## 6. Statistics Table and Math Verification

You can easily query computed statistics directly for tabular reporting in scientific manuscripts:

```python
from taylor_diagram import TaylorDiagram
import numpy as np

obs = np.random.randn(200) + 5
pred = obs * 0.92 + np.random.randn(200) * 0.25

td = TaylorDiagram(obs)
td.add_model(pred, label="Model_X")
td.plot()

# 1. Print formatted markdown-ready string table
print(td.get_stats_table())

# 2. Extract statistics dictionary for custom parsing
stats = td.models[0]
print(f"Algorithm: {stats['label']}")
print(f"Normalized Std: {stats['std']:.4f}")
print(f"Correlation: {stats['corr']:.4f}")
print(f"Centered RMSD: {stats['crmsd']:.4f}")
```

## 7. Integration with Scikit-Learn ML Pipeline

Easily evaluate and compare machine learning regression models:

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from taylor_diagram import TaylorDiagram

# 1. Generate synth regression dataset
X, y = make_regression(n_samples=500, n_features=10, noise=15.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train various estimators
models = {
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# 3. Predict on test set
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

# 4. Generate Taylor Diagram
td = TaylorDiagram(y_test, figsize=(8, 8), normalize=True)
markers = ["o", "s", "^"]
for (name, pred), marker in zip(predictions.items(), markers):
    td.add_model(pred, label=name, marker=marker)

td.plot(title="Machine Learning Regressors - Taylor Diagram Comparison")
td.savefig("ml_pipeline_comparison.png", dpi=300)
```