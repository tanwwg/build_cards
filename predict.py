import sys
import numpy as np
import tensorflow as tf
import cv2

MODEL_FILE = "build\\pokemon_cards.tflite"
LABELS_FILE = "build\\labels.txt"
IMG_SIZE = (224, 224)


def load_labels():
    with open(LABELS_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_image(filename):
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {filename}")

    # OpenCV loads as BGR, convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMG_SIZE)

    # Keep as float32 in 0-255 range
    image = image.astype(np.float32)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image>")
        sys.exit(1)

    image_file = sys.argv[1]
    labels = load_labels()

    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image = load_image(image_file)

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])[0]

    results = sorted(
        enumerate(predictions),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nImage: {image_file}\n")
    for index, confidence in results[:5]:
        print(f"{labels[index]:20s} {confidence * 100:6.2f}%")


if __name__ == "__main__":
    main()