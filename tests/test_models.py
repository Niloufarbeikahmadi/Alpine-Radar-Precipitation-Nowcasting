"""ConvLSTM architectures and the trained checkpoint (requires TensorFlow)."""

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from radar_nowcasting.models import MODEL_BUILDERS, build_model, load_trained_model  # noqa: E402
from radar_nowcasting.preprocessing import inverse_transform_predictions  # noqa: E402
from radar_nowcasting.utils import load_json, load_test_predictions  # noqa: E402

# Parameter counts of the study configurations on the 5 x 64 x 64 x 1 input
EXPECTED_PARAMS = {"light": 35_465, "moderate": 140_049, "unet": 5_758_273}


@pytest.mark.parametrize("architecture", sorted(MODEL_BUILDERS))
def test_output_shape(architecture):
    model = build_model(architecture, (5, 16, 16, 1), forecast_horizon=3)
    assert model.output_shape == (None, 3, 16, 16, 1)
    out = model.predict(np.zeros((2, 5, 16, 16, 1), np.float32), verbose=0)
    assert out.shape == (2, 3, 16, 16, 1)


@pytest.mark.parametrize("architecture", sorted(EXPECTED_PARAMS))
def test_parameter_counts(architecture):
    model = build_model(architecture, (5, 64, 64, 1), forecast_horizon=3)
    assert model.count_params() == EXPECTED_PARAMS[architecture]


def test_light_model_hyperparameters():
    model = build_model("light", (5, 64, 64, 1))
    convs = [layer for layer in model.layers if layer.__class__.__name__ == "ConvLSTM2D"]
    assert [c.filters for c in convs] == [16, 16, 8]
    assert all(c.kernel_regularizer.get_config()["l2"] == pytest.approx(5e-5) for c in convs)
    drops = [layer.rate for layer in model.layers if layer.__class__.__name__ == "Dropout"]
    assert drops == [0.25, 0.25, 0.25]
    opt = model.optimizer.get_config()
    assert opt["learning_rate"] == pytest.approx(1e-3) and opt["clipnorm"] == 1.0


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_model("transformer", (5, 16, 16, 1))


def test_trained_checkpoint_reproduces_reported_predictions(project_root):
    """The shipped weights regenerate the shipped test-set forecasts."""
    preds = load_test_predictions(project_root / "outputs/predictions/test_predictions.npz")
    scaler = load_json(project_root / "data/processed/scaler.json")
    model = load_trained_model(project_root / "outputs/models/best_nowcasting_model.h5")

    out = inverse_transform_predictions(model.predict(preds["X_test"], batch_size=4, verbose=0), scaler)
    np.testing.assert_allclose(out, preds["convlstm"], atol=1e-3)
