import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from taylor_diagram import TaylorDiagram, compute_taylor_stats


class TestComputeTaylorStats:
    """Unit tests for compute_taylor_stats function."""

    @pytest.mark.unit
    def test_perfect_prediction(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_taylor_stats(obs, obs, normalize=True)
        assert abs(stats["corr"] - 1.0) < 1e-10
        assert abs(stats["std"] - 1.0) < 1e-10
        assert abs(stats["crmsd"] - 0.0) < 1e-10

    @pytest.mark.unit
    def test_normalized_std(self):
        np.random.seed(42)
        obs = np.random.randn(200) + 5
        pred = obs * 2
        stats = compute_taylor_stats(obs, pred, normalize=True)
        assert abs(stats["std"] - 2.0) < 1e-10

    @pytest.mark.unit
    def test_non_normalized_std(self):
        np.random.seed(42)
        obs = np.random.randn(200) + 5
        stats = compute_taylor_stats(obs, obs, normalize=False)
        assert abs(stats["std"] - np.std(obs)) < 1e-10

    @pytest.mark.unit
    def test_correlation_range(self):
        np.random.seed(42)
        obs = np.random.randn(200)
        pred = np.random.randn(200)
        stats = compute_taylor_stats(obs, pred, normalize=True)
        assert -1.0 <= stats["corr"] <= 1.0

    @pytest.mark.unit
    def test_crmsd_non_negative(self):
        np.random.seed(42)
        obs = np.random.randn(200) + 5
        pred = np.random.randn(200) * 0.5 + 3
        stats = compute_taylor_stats(obs, pred, normalize=True)
        assert stats["crmsd"] >= 0

    @pytest.mark.unit
    def test_raw_stats_keys(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 2.1, 3.1])
        stats = compute_taylor_stats(obs, pred, normalize=True)
        assert "raw_std" in stats
        assert "raw_crmsd" in stats


class TestTaylorDiagram:
    """Unit tests for TaylorDiagram class."""

    @pytest.mark.unit
    def test_init(self):
        obs = np.random.randn(100) + 5
        td = TaylorDiagram(obs, figsize=(6, 6))
        assert td.obs_std > 0
        assert len(td.models) == 0

    @pytest.mark.unit
    def test_add_model(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        pred = obs + np.random.randn(100) * 0.3
        td = TaylorDiagram(obs)
        td.add_model(pred, label="TestModel", marker="o", color="blue")
        assert len(td.models) == 1
        assert td.models[0]["label"] == "TestModel"

    @pytest.mark.unit
    def test_add_multiple_models(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        td = TaylorDiagram(obs)
        for i in range(5):
            pred = obs + np.random.randn(100) * (0.1 + i * 0.2)
            td.add_model(pred, label=f"Model{i}")
        assert len(td.models) == 5

    @pytest.mark.unit
    def test_plot_no_error(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        pred = obs + np.random.randn(100) * 0.3
        td = TaylorDiagram(obs, figsize=(6, 6))
        td.add_model(pred, label="Test")
        td.plot(title="Test Diagram")

    @pytest.mark.unit
    def test_savefig(self, tmp_path):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        pred = obs + np.random.randn(100) * 0.3
        td = TaylorDiagram(obs, figsize=(6, 6))
        td.add_model(pred, label="Test")
        td.plot()
        out_path = str(tmp_path / "test_taylor.png")
        td.savefig(out_path, dpi=100)
        import os
        assert os.path.exists(out_path)

    @pytest.mark.unit
    def test_stats_table(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        pred = obs + np.random.randn(100) * 0.3
        td = TaylorDiagram(obs)
        td.add_model(pred, label="TestModel")
        table = td.get_stats_table()
        assert "TestModel" in table
        assert "Observed" in table

    @pytest.mark.unit
    def test_normalize_option(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        pred = obs * 1.5
        td_norm = TaylorDiagram(obs, normalize=True)
        td_norm.add_model(pred, label="M")
        td_raw = TaylorDiagram(obs, normalize=False)
        td_raw.add_model(pred, label="M")
        # Need to call plot() because standard deviation max limit is computed lazily during plot()
        td_norm.plot()
        td_raw.plot()
        assert abs(td_norm.models[0]["std"] - 1.5) < 1e-10
        assert abs(td_raw.models[0]["std"] - np.std(pred)) < 1e-10

    @pytest.mark.unit
    def test_dynamic_scaling(self):
        np.random.seed(42)
        obs = np.random.randn(100) + 5
        # Create a model prediction with huge standard deviation (4.0 times observed)
        pred = obs * 4.0
        td = TaylorDiagram(obs, normalize=True)
        td.add_model(pred, label="HugeModel")
        td.plot()
        # Verify that max_std_limit scaled up dynamically and covers the huge model
        assert td.max_std_limit >= 4.0
        assert td.max_std_limit < 5.5

    @pytest.mark.unit
    def test_negative_correlation(self):
        np.random.seed(42)
        obs = np.random.randn(100)
        # Create a prediction with negative correlation
        pred = -obs + np.random.randn(100) * 0.1
        td = TaylorDiagram(obs, corr_range=(-1, 1))
        td.add_model(pred, label="NegativeCorrModel")
        td.plot()
        # Verify the model has negative correlation
        assert td.models[0]["corr"] < 0
        # Check that it plotted without error
        assert hasattr(td, 'ax')


class TestE2E:
    """End-to-end tests for Taylor diagram workflow."""

    @pytest.mark.e2e
    def test_full_workflow(self, tmp_path):
        np.random.seed(42)
        obs = np.random.randn(200) + 5
        pred_a = obs + np.random.randn(200) * 0.3
        pred_b = obs * 0.8 + np.random.randn(200) * 0.5 + 1

        td = TaylorDiagram(obs, figsize=(8, 8))
        td.add_model(pred_a, label="Model A", marker="o", color="tab:blue")
        td.add_model(pred_b, label="Model B", marker="s", color="tab:orange")
        td.plot(title="E2E Test", show_crmsd=True)

        out_path = str(tmp_path / "e2e_taylor.png")
        td.savefig(out_path, dpi=150)

        import os
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

        table = td.get_stats_table()
        assert "Model A" in table
        assert "Model B" in table
        assert "Observed" in table

    @pytest.mark.e2e
    def test_multi_model_comparison(self, tmp_path):
        np.random.seed(42)
        obs = np.random.randn(300) * 2 + 10

        models = {
            "SVR": obs * 0.9 + np.random.randn(300) * 0.4,
            "RF": obs * 0.95 + np.random.randn(300) * 0.3,
            "XGBoost": obs * 0.98 + np.random.randn(300) * 0.2,
        }

        td = TaylorDiagram(obs, figsize=(9, 9))
        markers = ["o", "s", "^"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for (name, pred), mk, clr in zip(models.items(), markers, colors):
            td.add_model(pred, label=name, marker=mk, color=clr)

        td.plot(title="Multi-Model E2E")
        out_path = str(tmp_path / "e2e_multi.png")
        td.savefig(out_path, dpi=150)

        import os
        assert os.path.exists(out_path)
        assert len(td.models) == 3