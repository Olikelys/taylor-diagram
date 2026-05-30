#!/usr/bin/env python3
"""
Taylor Diagram - Multi-algorithm metric comparison visualization.

Provides a TaylorDiagram class that wraps matplotlib to create
publication-quality Taylor diagrams for comparing multiple model predictions
against observations. Three key metrics are displayed simultaneously:
  - Standard Deviation (radial distance)
  - Correlation Coefficient (angular position)
  - Centered RMS Difference (distance from reference point)

Usage:
    from taylor_diagram import TaylorDiagram

    td = TaylorDiagram(observed, figsize=(8, 8))
    td.add_model(pred1, label="Model A")
    td.add_model(pred2, label="Model B")
    td.plot()
    td.savefig("taylor_diagram.png", dpi=300)
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
import mpl_toolkits.axisartist.grid_finder as grid_finder

# Set high-quality journal publication defaults
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
})

# Journal-grade color palette families (Nature-inspired colorblind safe colors)
NATURE_PALETTE = [
    "#1F77B4",  # Steel Blue
    "#FF7F0E",  # Muted Orange
    "#2CA02C",  # Muted Green
    "#D62728",  # Muted Red
    "#9467BD",  # Muted Purple
    "#8C564B",  # Chestnut Brown
    "#E377C2",  # Muted Pink
    "#17BECF",  # Muted Teal
    "#BCBD22",  # Muted Olive
    "#7F7F7F"   # Medium Gray
]

class TaylorDiagram:
    """
    Create a Taylor diagram for multi-algorithm metric comparison.

    The Taylor diagram is a polar plot where:
    - Radial distance = standard deviation
    - Angular position = arccos(correlation coefficient)
    - Distance from reference point = centered RMS difference

    Parameters
    ----------
    observed : array-like
        1-D array of observed (reference) values.
    fig : matplotlib.figure.Figure, optional
        Existing figure to draw on. If None, a new figure is created.
    rect : int or tuple, optional
        Subplot specification for add_subplot. Default 111.
    figsize : tuple, optional
        Figure size in inches. Default (8, 8).
    normalize : bool, optional
        If True, normalize standard deviations by the observed std.
        Default True.
    corr_range : tuple, optional
        Correlation coefficient range (min, max). Default (0, 1).
    label_obs : str, optional
        Label for the observed reference point. Default "Observed".
    max_std : float, optional
        Manual maximum standard deviation limit for the plot radial axis.
        If None, dynamically calculated.
    """

    def __init__(self, observed, fig=None, rect=111, figsize=(8, 8),
                 normalize=True, corr_range=(0, 1), label_obs="Observed",
                 max_std=None):
        self.observed = np.asarray(observed, dtype=float)
        self.normalize = normalize
        self.corr_range = corr_range
        self.label_obs = label_obs
        self.max_std = max_std
        self.rect = rect

        self.obs_std = np.std(self.observed)
        self.models = []

        if fig is None:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = fig

        # Note: Axes are not initialized in __init__ to allow dynamic standard
        # deviation scaling depending on all models that will be added.
        # Axis setup is completed inside the plot() call.

    def _setup_axes(self, rect, max_std_limit):
        """Set up the polar axes for the Taylor diagram."""
        corr_min, corr_max = self.corr_range

        import warnings
        # Suppress MatplotlibDeprecationWarning for apply_theta_transforms
        warnings.filterwarnings("ignore", message=".*apply_theta_transforms.*")

        # Standard PolarAxes transform allows perfect alignment of coordinates in axisartist
        tr = PolarAxes.PolarTransform()

        # Dynamic correlation ticks (rlocs) depending on corr_range
        pos_rlocs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 1.0])
        if corr_min < 0:
            # Symmetrically extend to negative correlations
            neg_rlocs = -pos_rlocs[1:]
            rlocs = np.concatenate([neg_rlocs[::-1], pos_rlocs])
        else:
            rlocs = pos_rlocs

        rlocs = rlocs[(rlocs >= corr_min) & (rlocs <= corr_max)]
        tlocs = np.arccos(rlocs)
        gl = grid_finder.FixedLocator(tlocs)
        tf1 = grid_finder.DictFormatter(
            dict(zip(tlocs, [f"{r:.2f}" for r in rlocs]))
        )

        # Dynamic standard deviation ticks using MaxNLocator
        import matplotlib.ticker as ticker
        locator = ticker.MaxNLocator(nbins=5, prune=None)
        slocs = locator.tick_values(0, max_std_limit)
        slocs = slocs[(slocs >= 0) & (slocs <= max_std_limit)]

        # Safety fallback
        if len(slocs) < 2:
            slocs = np.linspace(0, max_std_limit, 4)

        tf2 = grid_finder.DictFormatter(
            dict(zip(slocs, [f"{s:.2f}" if max_std_limit < 1.0 else f"{s:.1f}" for s in slocs]))
        )
        gl2 = grid_finder.FixedLocator(slocs)

        grid_helper = floating_axes.GridHelperCurveLinear(
            tr,
            extremes=(np.arccos(corr_max), np.arccos(corr_min),
                      0, max_std_limit),
            grid_locator1=gl,
            grid_locator2=gl2,
            tick_formatter1=tf1,
            tick_formatter2=tf2,
        )

        self.ax = floating_axes.FloatingSubplot(
            self.fig, rect, grid_helper=grid_helper
        )
        self.fig.add_subplot(self.ax)

        self.ax_aux = self.ax.get_aux_axes(tr)

        # ----------------------------------------------------------------------
        # Configure correct axes visibility and labelling for Taylor diagrams:
        # - "bottom" (r = r_min = 0) is the origin. It should be hidden.
        # - "top" (r = r_max) is the curved outer arc showing Correlation.
        # - "left" (theta = theta_max) is the vertical line at theta_max.
        # - "right" (theta = theta_min) is the bottom horizontal axis showing Standard Deviation.
        # ----------------------------------------------------------------------
        self.ax.axis["bottom"].set_visible(False)
        self.ax.axis["top"].set_visible(True)
        self.ax.axis["left"].set_visible(True)
        self.ax.axis["right"].set_visible(True)

        # Standard Deviation labels along bottom horizontal line ("right")
        self.ax.axis["right"].label.set_text("Standard Deviation")
        self.ax.axis["right"].label.set_fontsize(11)
        self.ax.axis["right"].label.set_fontweight("bold")
        self.ax.axis["right"].toggle(all=True)

        # Correlation labels along the curved outer arc ("top")
        self.ax.axis["top"].label.set_text("Correlation Coefficient")
        self.ax.axis["top"].label.set_fontsize(11)
        self.ax.axis["top"].label.set_fontweight("bold")
        self.ax.axis["top"].toggle(all=True)

        # Keep vertical line clean by hiding ticklabels on "left" axis
        self.ax.axis["left"].toggle(ticklabels=False, ticks=False)

        # Premium grid styling
        self.ax.grid(True, linestyle=":", color="#d0d0d0", alpha=0.8, linewidth=0.8)
        self.max_std_limit = max_std_limit

    def _compute_stats(self, predicted):
        """Compute Taylor diagram statistics for a prediction array."""
        predicted = np.asarray(predicted, dtype=float)
        pred_std = np.std(predicted)
        corr = np.corrcoef(self.observed, predicted)[0, 1]

        obs_mean = np.mean(self.observed)
        pred_mean = np.mean(predicted)
        crmsd = np.sqrt(np.mean((predicted - pred_mean - (self.observed - obs_mean)) ** 2))

        if self.normalize and self.obs_std > 0:
            norm_std = pred_std / self.obs_std
            norm_crmsd = crmsd / self.obs_std
        else:
            norm_std = pred_std
            norm_crmsd = crmsd

        return {
            "std": norm_std,
            "corr": corr,
            "crmsd": norm_crmsd,
            "raw_std": pred_std,
            "raw_crmsd": crmsd,
        }

    def add_model(self, predicted, label=None, marker="o", color=None,
                  markersize=10, **kwargs):
        """
        Add a model prediction to the diagram.

        Parameters
        ----------
        predicted : array-like
            1-D array of predicted values.
        label : str, optional
            Model label for the legend.
        marker : str, optional
            Marker style. Default "o".
        color : str or tuple, optional
            Marker color. If None, auto-selected from Nature color family.
        markersize : float, optional
            Marker size. Default 10.
        """
        stats = self._compute_stats(predicted)
        
        # Auto-select color from high-impact journal color palette
        if color is None:
            color_index = len(self.models) % len(NATURE_PALETTE)
            color = NATURE_PALETTE[color_index]

        stats.update({
            "label": label,
            "marker": marker,
            "color": color,
            "markersize": markersize,
            "kwargs": kwargs,
        })
        self.models.append(stats)

    def plot(self, show_crmsd=True, crmsd_levels=None, crmsd_color="gray",
             crmsd_style="--", crmsd_alpha=0.5, obs_marker="*",
             obs_color="black", obs_markersize=15, title=None,
             title_fontsize=14, legend=True, legend_kwargs=None):
        """
        Render the Taylor diagram.

        Parameters
        ----------
        show_crmsd : bool, optional
            Whether to draw centered RMS difference contours. Default True.
        crmsd_levels : array-like, optional
            Specific CRMSD contour levels. If None, auto-computed.
        crmsd_color : str, optional
            CRMSD contour color. Default "gray".
        crmsd_style : str, optional
            CRMSD contour linestyle. Default "--".
        crmsd_alpha : float, optional
            CRMSD contour alpha. Default 0.5.
        obs_marker : str, optional
            Observed point marker. Default "*".
        obs_color : str, optional
            Observed point color. Default "black".
        obs_markersize : float, optional
            Observed point size. Default 15.
        title : str, optional
            Diagram title.
        title_fontsize : float, optional
            Title font size. Default 14.
        legend : bool, optional
            Whether to show legend. Default True.
        legend_kwargs : dict, optional
            Additional kwargs for legend.
        """
        # Complete delayed axes setup using computed standard deviation range
        if not hasattr(self, 'ax'):
            max_std = 1.0 if self.normalize else self.obs_std
            if self.models:
                max_model_std = max(m["std"] for m in self.models)
                max_std = max(max_std, max_model_std)

            if self.max_std is not None:
                max_std_limit = self.max_std
            else:
                max_std_limit = max_std * 1.15
                if self.normalize:
                    max_std_limit = max(1.5, max_std_limit)
                else:
                    max_std_limit = max(self.obs_std * 1.5, max_std_limit)

            self._setup_axes(self.rect, max_std_limit)

        obs_norm_std = 1.0 if self.normalize else self.obs_std

        # Plot observed reference point
        theta_obs = 0
        self.ax_aux.plot(theta_obs, obs_norm_std, marker=obs_marker,
                         color=obs_color, markersize=obs_markersize,
                         zorder=5, label=self.label_obs)

        # Plot centered RMS difference contours
        if show_crmsd:
            self._draw_crmsd_contours(obs_norm_std, crmsd_levels,
                                       crmsd_color, crmsd_style, crmsd_alpha)

        # Plot each model point
        for m in self.models:
            theta = np.arccos(np.clip(m["corr"], -1, 1))
            self.ax_aux.plot(theta, m["std"], marker=m["marker"],
                             color=m["color"], markersize=m["markersize"],
                             label=m["label"], zorder=4, **m["kwargs"])

        # Display polished legend
        if legend and any(m["label"] for m in self.models):
            lk = {
                "loc": "upper right",
                "bbox_to_anchor": (1.1, 1.1) if self.corr_range[0] >= 0 else (1.2, 1.1),
                "fontsize": 9,
                "frameon": False
            }
            if legend_kwargs:
                lk.update(legend_kwargs)
            self.ax_aux.legend(**lk)

        # Set title
        if title:
            self.ax.set_title(title, fontsize=title_fontsize, pad=20, fontweight="bold")

    def _draw_crmsd_contours(self, obs_std, levels, color, style, alpha):
        """Draw centered RMS difference contour arcs mathematically correctly."""
        if levels is None:
            # Dynamically determine reasonable CRMSD contour intervals
            max_model_std = max(m["std"] for m in self.models) if self.models else 1.0
            max_crmsd = obs_std + max_model_std
            
            import matplotlib.ticker as ticker
            locator = ticker.MaxNLocator(nbins=5)
            levels = locator.tick_values(0.1, max_crmsd)
            levels = levels[levels > 0]
            if len(levels) == 0:
                levels = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

        for level in levels:
            # Parametric representation of a circle centered at (obs_std, 0)
            # x = obs_std + level * cos(phi)
            # y = level * sin(phi)
            # theta = arctan2(y, x), r = sqrt(x^2 + y^2)
            # phi goes 0 to pi representing the upper half plane (y >= 0)
            phi = np.linspace(0, np.pi, 150)
            x = obs_std + level * np.cos(phi)
            y = level * np.sin(phi)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

            # Filter points within plot boundaries
            corr_min, corr_max = self.corr_range
            mask = (theta >= np.arccos(corr_max)) & \
                   (theta <= np.arccos(corr_min)) & \
                   (r <= self.max_std_limit)

            if np.any(mask):
                self.ax_aux.plot(theta[mask], r[mask], linestyle=style,
                                 color=color, alpha=alpha, linewidth=0.8)

    def savefig(self, filename, dpi=300, bbox_inches="tight", **kwargs):
        """Save the diagram to a file."""
        self.fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)

    def get_stats_table(self):
        """
        Return a formatted string table of all model statistics.

        Returns
        -------
        str
            Formatted table with columns: Model, Std, Corr, CRMSD.
        """
        header = f"{'Model':<20} {'Std':>8} {'Corr':>8} {'CRMSD':>8}"
        sep = "-" * len(header)
        lines = [header, sep]
        obs_norm_std = 1.0 if self.normalize else self.obs_std
        lines.append(f"{self.label_obs:<20} {obs_norm_std:>8.4f} "
                     f"{'1.0000':>8} {'0.0000':>8}")
        for m in self.models:
            lines.append(f"{m['label']:<20} {m['std']:>8.4f} "
                         f"{m['corr']:>8.4f} {m['crmsd']:>8.4f}")
        return "\n".join(lines)


def compute_taylor_stats(observed, predicted, normalize=True):
    """
    Compute Taylor diagram statistics for a single model.

    Parameters
    ----------
    observed : array-like
        Observed (reference) values.
    predicted : array-like
        Predicted values.
    normalize : bool, optional
        Normalize by observed std. Default True.

    Returns
    -------
    dict
        Dictionary with keys: std, corr, crmsd, raw_std, raw_crmsd.
    """
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    obs_std = np.std(observed)
    pred_std = np.std(predicted)
    corr = np.corrcoef(observed, predicted)[0, 1]

    obs_mean = np.mean(observed)
    pred_mean = np.mean(predicted)
    crmsd = np.sqrt(np.mean((predicted - pred_mean - (observed - obs_mean)) ** 2))

    if normalize and obs_std > 0:
        norm_std = pred_std / obs_std
        norm_crmsd = crmsd / obs_std
    else:
        norm_std = pred_std
        norm_crmsd = crmsd

    return {
        "std": norm_std,
        "corr": corr,
        "crmsd": norm_crmsd,
        "raw_std": pred_std,
        "raw_crmsd": crmsd,
    }


def plot_taylor_skillmetrics(sdev, crmsd, ccoef, markerLabel=None,
                             figsize=(8, 8), dpi=150, save_path=None,
                             **kwargs):
    """
    Plot Taylor diagram using the SkillMetrics library.

    This is a convenience wrapper around SkillMetrics.taylor_diagram()
    for users who prefer the SkillMetrics API.

    Parameters
    ----------
    sdev : array-like
        Standard deviations. First element is the reference.
    crmsd : array-like
        Centered RMS differences. First element is the reference (0).
    ccoef : array-like
        Correlation coefficients. First element is the reference (1).
    markerLabel : list of str, optional
        Labels for each data point.
    figsize : tuple, optional
        Figure size. Default (8, 8).
    dpi : int, optional
        Figure DPI. Default 150.
    save_path : str, optional
        File path to save the figure.
    **kwargs
        Additional arguments passed to sm.taylor_diagram().

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    try:
        import skill_metrics as sm
    except ImportError:
        raise ImportError(
            "SkillMetrics package is required. "
            "Install with: pip install SkillMetrics"
        )

    plt.close("all")
    fig = plt.figure(figsize=figsize, dpi=dpi)

    plot_kwargs = {}
    if markerLabel is not None:
        plot_kwargs["markerLabel"] = markerLabel
    plot_kwargs.update(kwargs)

    sm.taylor_diagram(sdev, crmsd, ccoef, **plot_kwargs)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig


if __name__ == "__main__":
    np.random.seed(42)
    obs = np.random.randn(200) + 5

    pred1 = obs + np.random.randn(200) * 0.3
    pred2 = obs * 0.8 + np.random.randn(200) * 0.5 + 1
    pred3 = obs * 0.6 + np.random.randn(200) * 1.0 + 2

    td = TaylorDiagram(obs, figsize=(8, 8))
    td.add_model(pred1, label="Model A", marker="o")
    td.add_model(pred2, label="Model B", marker="s")
    td.add_model(pred3, label="Model C", marker="^")
    td.plot(title="Taylor Diagram - Multi-Model Comparison")
    td.savefig("taylor_diagram_example.png", dpi=150)
    print(td.get_stats_table())
    plt.show()