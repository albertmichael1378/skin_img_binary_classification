import tensorflow as tf


GRAD_CAM_LAYER = "top_activation"


def build_backbone(
    *,
    input_shape: tuple[int, int, int] | list[int],
    input_tensor: tf.Tensor,
    weights: str | None,
    include_top: bool,
) -> tf.keras.Model:
    """Build EfficientNetV2-S with its native input preprocessing enabled."""
    return tf.keras.applications.EfficientNetV2S(
        input_shape=tuple(input_shape),
        input_tensor=input_tensor,
        weights=weights,
        include_top=include_top,
        include_preprocessing=True,
    )
