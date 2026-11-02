import base64
from io import BytesIO

import numpy as np


def grad_cam(
    model,
    image: np.ndarray,
    class_index: int,
    layer_name: str = "block5_conv3",
) -> np.ndarray:
    """Create a normalized Grad-CAM map for one class and one RGB image."""
    import tensorflow as tf

    if class_index not in (0, 1):
        raise ValueError(f"Expected binary class index 0 or 1, got {class_index}")

    try:
        convolution_layer = model.get_layer(layer_name)
    except ValueError as error:
        raise ValueError(
            f"Grad-CAM layer {layer_name!r} was not found in the model"
        ) from error

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[convolution_layer.output, model.output],
    )
    image_tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        convolution_output, predictions = grad_model(image_tensor, training=False)
        output_count = predictions.shape[-1]
        if output_count == 1:
            probability = predictions[:, 0]
            class_score = probability if class_index == 1 else 1.0 - probability
        elif output_count == 2:
            class_score = predictions[:, class_index]
        else:
            raise ValueError(
                f"Expected one or two binary class outputs, got {output_count}"
            )

    gradients = tape.gradient(class_score, convolution_output)
    if gradients is None:
        raise RuntimeError(
            f"No gradient connects model output to layer {layer_name!r}"
        )

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.nn.relu(
        tf.reduce_sum(
            convolution_output * channel_weights[:, None, None, :],
            axis=-1,
        )
    )[0]
    heatmap /= tf.reduce_max(heatmap) + tf.keras.backend.epsilon()
    return heatmap.numpy()


def encode_grad_cam_overlay(image: np.ndarray, heatmap: np.ndarray) -> str:
    """Resize a Grad-CAM map, blend it with an RGB image, and return JPEG base64."""
    from PIL import Image

    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")

    source_image = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))
    resized_heatmap = Image.fromarray(
        np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    ).resize(source_image.size, Image.Resampling.BILINEAR)
    heatmap_values = np.asarray(resized_heatmap, dtype=np.float32) / 255.0

    heatmap_rgb = np.stack(
        [
            np.clip(1.5 - np.abs(4 * heatmap_values - 3), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap_values - 2), 0, 1),
            np.clip(1.5 - np.abs(4 * heatmap_values - 1), 0, 1),
        ],
        axis=-1,
    )
    blended = (
        np.asarray(source_image, dtype=np.float32) * 0.6
        + heatmap_rgb * 255 * 0.4
    )
    overlay = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    output = BytesIO()
    overlay.save(output, format="JPEG")
    return base64.b64encode(output.getvalue()).decode("ascii")
