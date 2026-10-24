"""
Task 2 - Deep-learning architectures for radar precipitation nowcasting.

Three ConvLSTM variants were explored during the study, from heavy to light.
They are kept side by side so the architecture search is reproducible:

==============  ==========================================  =================
Key             Description                                 Status
==============  ==========================================  =================
``"unet"``      U-Net ConvLSTM encoder-decoder, 64-128       Rejected
                filters, spatial down/up-sampling, skips     (over-fitting)
``"moderate"``  3-block ConvLSTM 32-32-16, L2 = 1e-4         Intermediate
``"light"``     3-block ConvLSTM 16-16-8, L2 = 5e-5          **Adopted**
==============  ==========================================  =================

All models map an input sequence ``(T_in, H, W, 1)`` of z-scored
``log(1 + R)`` rainfall to the next ``T_out`` frames ``(T_out, H, W, 1)`` with a
linear 1x1 convolution head, and are compiled with Adam (lr = 1e-3,
clipnorm = 1.0) on an MSE loss.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    ConvLSTM2D,
    Dropout,
    Input,
    TimeDistributed,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .config import TrainingConfig

InputShape = Tuple[int, int, int, int]


def _compile(model: Model, cfg: TrainingConfig = TrainingConfig()) -> Model:
    optimizer = Adam(learning_rate=cfg.learning_rate, clipnorm=cfg.clipnorm)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


# ===========================================================================
# 1) Heavy U-Net ConvLSTM  (rejected: ~0.5 M+ parameters, over-fitted)
# ===========================================================================
def build_unet_convlstm(input_shape: InputShape, forecast_horizon: int = 3) -> Model:
    """U-Net style ConvLSTM encoder-decoder with skip connections."""
    inputs = Input(shape=input_shape, name="input_sequence")

    def convlstm(filters: int, name: str, strides=(1, 1)):
        return ConvLSTM2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            return_sequences=True,
            activation="tanh",
            name=name,
        )

    # Encoder
    conv1 = convlstm(64, "enc_conv1")(inputs)
    conv1 = BatchNormalization(name="enc_bn1")(conv1)

    conv2 = convlstm(128, "enc_conv2", strides=(2, 2))(conv1)  # downsample
    conv2 = BatchNormalization(name="enc_bn2")(conv2)

    conv3 = convlstm(128, "enc_conv3", strides=(2, 2))(conv2)  # downsample
    conv3 = BatchNormalization(name="enc_bn3")(conv3)

    # Bottleneck
    conv4 = convlstm(128, "bottleneck_conv")(conv3)
    conv4 = BatchNormalization(name="bottleneck_bn")(conv4)

    # Decoder (upsampling + skip connections)
    up1 = TimeDistributed(UpSampling2D(size=(2, 2)), name="dec_up1")(conv4)
    up1 = Concatenate(axis=-1, name="dec_concat1")([up1, conv2])
    up1 = convlstm(128, "dec_conv1")(up1)
    up1 = BatchNormalization(name="dec_bn1")(up1)

    up2 = TimeDistributed(UpSampling2D(size=(2, 2)), name="dec_up2")(up1)
    up2 = Concatenate(axis=-1, name="dec_concat2")([up2, conv1])
    up2 = convlstm(64, "dec_conv2")(up2)
    up2 = BatchNormalization(name="dec_bn2")(up2)

    # Keep the last `forecast_horizon` time steps -> (None, T_out, H, W, 64)
    forecast_frames = up2[:, -forecast_horizon:, ...]

    output = TimeDistributed(
        Conv2D(filters=1, kernel_size=(1, 1), activation="linear", padding="same"),
        name="rainfall_output",
    )(forecast_frames)

    model = Model(inputs=inputs, outputs=output, name="UNetConvLSTMNowcasting")
    return _compile(model)


# ===========================================================================
# 2) & 3) Stacked ConvLSTM (no spatial down/up-sampling)
# ===========================================================================
def _build_stacked_convlstm(
    input_shape: InputShape,
    forecast_horizon: int,
    filters: Tuple[int, int, int],
    l2_strength: float,
    dropout_rate: float,
) -> Model:
    """Three regularised ConvLSTM blocks (BatchNorm + Dropout) and a 1x1 linear head."""
    l2_reg = regularizers.l2(l2_strength)
    inputs = Input(shape=input_shape, name="input_sequence")

    x = inputs
    for block, n_filters in enumerate(filters, start=1):
        x = ConvLSTM2D(
            filters=n_filters,
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="tanh",
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            bias_regularizer=l2_reg,
            name=f"convlstm_block{block}",
        )(x)
        x = BatchNormalization(name=f"bn_block{block}")(x)
        x = Dropout(dropout_rate, name=f"drop_block{block}")(x)

    # Keep the last `forecast_horizon` time steps
    forecast_frames = x[:, -forecast_horizon:, ...]

    # 1x1 Conv2D per frame, linear activation (targets are z-scored log(1+R))
    output = TimeDistributed(
        Conv2D(
            filters=1,
            kernel_size=(1, 1),
            activation="linear",
            padding="same",
            name="rainfall_output",
        ),
        name="td_output",
    )(forecast_frames)

    model = Model(inputs=inputs, outputs=output, name="LightConvLSTMNowcasting")
    return _compile(model)


def build_moderate_convlstm(input_shape: InputShape, forecast_horizon: int = 3) -> Model:
    """Moderate ConvLSTM: filters 32-32-16, L2 = 1e-4, dropout = 0.25."""
    return _build_stacked_convlstm(
        input_shape,
        forecast_horizon,
        filters=(32, 32, 16),
        l2_strength=1e-4,
        dropout_rate=0.25,
    )


def build_light_convlstm(input_shape: InputShape, forecast_horizon: int = 3) -> Model:
    """Light ConvLSTM (adopted): filters 16-16-8, L2 = 5e-5, dropout = 0.25."""
    return _build_stacked_convlstm(
        input_shape,
        forecast_horizon,
        filters=(16, 16, 8),
        l2_strength=5e-5,
        dropout_rate=0.25,
    )


# ===========================================================================
# Registry
# ===========================================================================
MODEL_BUILDERS: Dict[str, Callable[[InputShape, int], Model]] = {
    "unet": build_unet_convlstm,
    "moderate": build_moderate_convlstm,
    "light": build_light_convlstm,
}


def build_model(architecture: str, input_shape: InputShape, forecast_horizon: int = 3) -> Model:
    """Build one of the registered architectures by key (``unet``/``moderate``/``light``)."""
    try:
        builder = MODEL_BUILDERS[architecture]
    except KeyError as exc:
        raise ValueError(f"Unknown architecture '{architecture}'. Choose from {sorted(MODEL_BUILDERS)}.") from exc
    return builder(input_shape, forecast_horizon)


def load_trained_model(
    weights_path,
    architecture: str = "light",
    input_shape: InputShape = (5, 64, 64, 1),
    forecast_horizon: int = 3,
) -> Model:
    """Rebuild an architecture and load trained weights from a ``.h5`` checkpoint.

    Rebuilding from code (rather than ``keras.models.load_model``) keeps the
    checkpoint loadable across Keras versions, because the time-slice
    operation in the graph is re-created natively.
    """
    model = build_model(architecture, input_shape, forecast_horizon)
    model.load_weights(str(weights_path))
    return model
