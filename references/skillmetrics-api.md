# SkillMetrics API Reference

## Installation

```bash
pip install SkillMetrics
```

## Core Functions

### taylor_statistics(pred, ref, data_type)

Compute Taylor diagram statistics for a single model prediction.

**Parameters:**
- `pred` (array): Predicted values
- `ref` (array): Reference (observed) values
- `data_type` (str): 'data' for raw data arrays

**Returns:**
- `dict` with keys:
  - `sdev`: [obs_std, pred_std]
  - `crmsd`: [0, crmsd_value]
  - `ccoef`: [1, corr_value]

### taylor_diagram(sdev, crmsd, ccoef, **kwargs)

Plot a Taylor diagram using pre-computed statistics.

**Required Parameters:**
- `sdev` (array): Standard deviations, first element is reference
- `crmsd` (array): Centered RMS differences, first element is 0
- `ccoef` (array): Correlation coefficients, first element is 1

**Key Optional Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `markerLabel` | list | None | Labels for each data point |
| `markerLabelColor` | str | 'k' | Color of marker labels |
| `markerLegend` | str | 'off' | 'on' to show legend |
| `markerDisplayed` | str | 'marker' | 'marker', 'colorBar', or 'numberLabel' |
| `markerSize` | float | 10 | Size of markers |
| `titleRMS` | str | 'on' | 'on'/'off' for RMS axis title |
| `titleRMSDangle` | float | 35.0 | Angle for RMSD label |
| `tickRMS` | array | auto | RMS tick values |
| `tickRMSangle` | float | 115.0 | Angle for RMS ticks |
| `colRMS` | str | 'gray' | RMS contour color |
| `styleRMS` | str | ':' | RMS contour linestyle |
| `widthRMS` | float | 2.0 | RMS contour linewidth |
| `tickSTD` | array | auto | STD tick values |
| `axismax` | float | auto | Maximum axis value |
| `colSTD` | str | 'b' | STD axis color |
| `styleSTD` | str | '-.' | STD axis linestyle |
| `widthSTD` | float | 1.0 | STD axis linewidth |
| `titleSTD` | str | 'on' | 'on'/'off' for STD axis title |
| `colCOR` | str | 'k' | Correlation axis color |
| `styleCOR` | str | '--' | Correlation axis linestyle |
| `widthCOR` | float | 1.0 | Correlation axis linewidth |
| `titleCOR` | str | 'on' | 'on'/'off' for COR axis title |
| `titleOBS` | str | 'Observation' | Label for reference point |
| `colOBS` | str | 'k' | Reference point color |
| `styleOBS` | str | '-' | Reference line style |
| `widthOBS` | float | 2.0 | Reference line width |
| `alpha` | float | 0.0 | Background transparency |
| `cmapzdata` | array | None | Data for colorbar mapping |
| `titleColorbar` | str | '' | Colorbar title |
| `locationColorBar` | str | 'EastOutside' | Colorbar position |

### Other Useful Functions

- `centered_rms_dev(pred, ref)`: Compute CRMSD
- `rmsd(pred, ref)`: Compute RMSD
- `bias(pred, ref)`: Compute bias
- `nash_sutcliffe_eff(pred, ref)`: Compute NSE
- `kling_gupta_eff09(pred, ref)`: Compute KGE (2009)
- `kling_gupta_eff12(pred, ref)`: Compute KGE (2012)

## Typical Workflow

```python
import skill_metrics as sm
import numpy as np

# 1. Compute statistics for each model
taylor_stats1 = sm.taylor_statistics(pred1, ref, 'data')
taylor_stats2 = sm.taylor_statistics(pred2, ref, 'data')
taylor_stats3 = sm.taylor_statistics(pred3, ref, 'data')

# 2. Assemble arrays (first element = reference)
sdev = np.array([
    taylor_stats1['sdev'][0],   # obs std
    taylor_stats1['sdev'][1],   # model1 std
    taylor_stats2['sdev'][1],   # model2 std
    taylor_stats3['sdev'][1],   # model3 std
])
crmsd = np.array([
    taylor_stats1['crmsd'][0],  # 0 (reference)
    taylor_stats1['crmsd'][1],  # model1 crmsd
    taylor_stats2['crmsd'][1],  # model2 crmsd
    taylor_stats3['crmsd'][1],  # model3 crmsd
])
ccoef = np.array([
    taylor_stats1['ccoef'][0],  # 1 (reference)
    taylor_stats1['ccoef'][1],  # model1 corr
    taylor_stats2['ccoef'][1],  # model2 corr
    taylor_stats3['ccoef'][1],  # model3 corr
])

# 3. Plot
label = ['Obs', 'Model1', 'Model2', 'Model3']
sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=label)
```

## Important Notes

1. **Array order**: First element must always be the reference (observed) data
2. **Reference values**: sdev[0]=obs_std, crmsd[0]=0, ccoef[0]=1
3. **Data loading**: For CSV data, use `load_data` from `skill_metrics.load_data`
4. **Package rename**: Formerly `skill_metrics`, now `SkillMetrics` on PyPI

## GitHub Repository

https://github.com/PeterRochford/SkillMetrics

## Citation

Peter A. Rochford (2016) SkillMetrics: A Python package for calculating the skill of model predictions against observations, https://github.com/PeterRochford/SkillMetrics