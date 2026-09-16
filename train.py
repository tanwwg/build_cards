from pathlib import Path
import tensorflow as tf
from tensorflow import keras

# -----------------------------
# Settings
# -----------------------------

DATA_DIR = "augment"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

MODEL_FILE = "build\\pokemon_cards.tflite"
LABELS_FILE = "build\\labels.txt"

# -----------------------------
# Load dataset
# -----------------------------

train_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names

print()
print("Classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
print()

# Improve input pipeline performance
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# -----------------------------
# Pretrained MobileNetV3Small
# -----------------------------

base_model = keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
)

# Keep pretrained layers frozen.
# We only train our new classification layer.
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

# MobileNetV3 includes input preprocessing by default.
x = base_model(inputs, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(
    len(class_names),
    activation="softmax"
)(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# -----------------------------
# Train
# -----------------------------

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
)

# -----------------------------
# Save labels
# -----------------------------

with open(LABELS_FILE, "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print(f"\nSaved labels to {LABELS_FILE}")

# -----------------------------
# Convert to TensorFlow Lite
# -----------------------------

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(MODEL_FILE, "wb") as f:
    f.write(tflite_model)

print(f"Saved model to {MODEL_FILE}")

# -----------------------------
# Finished
# -----------------------------

print("\nDone!")
print(f"Classes: {len(class_names)}")
print(f"Model:   {MODEL_FILE}")
print(f"Labels:  {LABELS_FILE}")
