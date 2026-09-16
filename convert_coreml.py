import coremltools as ct
import tensorflow as tf

model = tf.keras.models.load_model("build/pokemon_cards.keras", compile=False)

with open("build/labels.txt", encoding="utf-8") as f:
    class_labels = f.read().splitlines()

if not class_labels or any(not label.strip() for label in class_labels):
    raise ValueError("build/labels.txt must contain one non-empty label per line")
if len(class_labels) != model.output_shape[-1]:
    raise ValueError("Label count must match the model's output class count")

# Core ML image inputs use a single image per prediction.
input_shape = (1, *model.input_shape[1:])

coreml_model = ct.convert(
    model,
    source="tensorflow",
    inputs=[
        ct.ImageType(
            # Omit name so Core ML uses the model's single input automatically.
            shape=input_shape,
            color_layout=ct.colorlayout.RGB,
            channel_first=False,
            # train.py's MobileNetV3 already rescales raw [0, 255] pixels.
            scale=1.0,
        )
    ],
    # Preserve train.py's class order and embed labels in the classifier.
    classifier_config=ct.ClassifierConfig(class_labels),
)

coreml_model.save("build/pokemon_cards.mlpackage")
