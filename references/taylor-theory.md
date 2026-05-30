# Taylor Diagram Theory

## Overview

The Taylor diagram, invented by Karl E. Taylor in 2001, is a polar coordinate visualization that simultaneously displays three statistical metrics for comparing model predictions against observations:

1. **Standard Deviation (Std)** - Radial distance from origin
2. **Correlation Coefficient (R)** - Angular position (arccos(R))
3. **Centered RMS Difference (CRMSD)** - Distance from the reference point

These three metrics satisfy the fundamental relationship:

```
CRMSD² = σ_obs² + σ_pred² - 2·σ_obs·σ_pred·R
```

This is the Law of Cosines applied in the Taylor diagram geometry, which is why the three metrics can be represented on a single 2D diagram.

## Metric Interpretation

### Standard Deviation (σ)
- Measures the amplitude of variations
- Radial distance from the origin in the diagram
- If normalized: σ_obs = 1.0 (reference)
- σ_pred close to 1.0 → model captures variance well
- σ_pred < 1.0 → model is too smooth (underestimates variability)
- σ_pred > 1.0 → model is too variable (overestimates variability)

### Correlation Coefficient (R)
- Measures pattern/phase similarity
- Angular position: θ = arccos(R)
- R = 1.0 → θ = 0° (rightmost position, perfect correlation)
- R = 0.0 → θ = 90° (perpendicular, no correlation)
- R close to 1.0 → model captures the pattern well

### Centered RMS Difference (E')
- Measures overall error after removing mean bias
- Geometric distance from the reference point (Observed) in the diagram
- E' = 0 → perfect match (model point coincides with observed point)
- Smaller E' → better overall model performance
- **Important**: This is NOT the same as RMSE. CRMSD removes the mean bias component:
  ```
  CRMSD = sqrt(mean((pred - mean(pred) - (obs - mean(obs)))²))
  ```

## Reading a Taylor Diagram

1. **Reference point** (typically a star ★): Represents the observed data
2. **Model points** (various markers): Each represents a different model/prediction
3. **Radial lines from origin**: Lines of constant standard deviation
4. **Angular lines from origin**: Lines of constant correlation
5. **Arcs centered on reference point**: Lines of constant CRMSD

**Key insight**: A model point closer to the reference point indicates better overall performance (lower CRMSD). The position also reveals *why* a model performs well or poorly:
- Close in radius → similar variance
- Small angle → high correlation
- Close in distance → low centered error

## When to Use Taylor Diagrams

- Comparing multiple models/algorithms on the same dataset
- Model intercomparison studies (climate, hydrology, air quality)
- Evaluating regression/prediction model performance
- Complementing traditional metrics (RMSE, R², MAE) with a visual summary
- Publication figures where "one diagram is worth a thousand words"

## When NOT to Use Taylor Diagrams

- Classification tasks (Taylor diagrams are for continuous predictions)
- When mean bias is the primary concern (CRMSD removes bias; use Target diagrams instead)
- When only 1-2 models are compared (a simple table may suffice)
- When the audience is unfamiliar with the diagram (consider adding annotations)

## Normalization

When `normalize=True` (default), all standard deviations are divided by σ_obs:
- The observed point appears at radius = 1.0
- Model std values become ratios (σ_pred / σ_obs)
- This makes diagrams comparable across different datasets

## Correlation Range

- **(0, 1)**: Default. Shows only positive correlations (right half). More common in practice.
- **(-1, 1)**: Full range. Shows both positive and negative correlations. Useful when negative correlations are expected.

## References

- Taylor, K. E. (2001). Summarizing multiple aspects of model performance in a single diagram. *J. Geophys. Res.*, 106(D7), 7183–7192. doi:10.1029/2000JD900719
- Taylor, K. E. (2001). Taylor Diagram Primer. PCMDI Report.