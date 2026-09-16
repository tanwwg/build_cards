import cv2
import albumentations as A
from pathlib import Path

transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.15,
        rotate_limit=10,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.8
    ),

    A.Perspective(
        scale=(0.02, 0.08),
        p=0.5
    ),

    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=5),
    ], p=0.3),

    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.6
    ),

    A.HueSaturationValue(
        hue_shift_limit=5,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3
    ),

    A.GaussNoise(
        std_range=(0.01, 0.05),
        p=0.25
    ),

    A.ImageCompression(
        quality_range=(60, 95),
        p=0.3
    ),
])


def augment_card(source_file, output_dir, count=100):
    source_file = Path(source_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(source_file))
    if image is None:
        raise ValueError(f"Failed to read source image: {source_file}")

    print(f"Generating {count} augmentations from {source_file} into {output_dir}", flush=True)

    original_file = output_dir / f"{source_file.stem}_0000.jpg"
    if not cv2.imwrite(str(original_file), image):
        raise OSError(f"Failed to save original image: {original_file}")
    print(f"Saved unaugmented original: {original_file}", flush=True)

    for i in range(1, count + 1):
        augmented = transform(image=image)["image"]

        output_file = output_dir / f"{source_file.stem}_{i:04d}.jpg"
        if not cv2.imwrite(str(output_file), augmented):
            raise OSError(f"Failed to save augmented image: {output_file}")
        print(f"[{i}/{count}] Saved {output_file}", flush=True)

    print(f"Finished saving 1 original and {count} augmentations in {output_dir}", flush=True)


if __name__ == "__main__":
    images_dir = Path("images")
    output_dir = Path("augment")
    source_files = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )

    print(f"Found {len(source_files)} PNG files in {images_dir}", flush=True)
    for index, source_file in enumerate(source_files, start=1):
        print(f"Processing file {index}/{len(source_files)}: {source_file.name}", flush=True)
        augment_card(source_file, output_dir / source_file.stem, count=200)

    print(f"Finished processing {len(source_files)} PNG files into {output_dir}", flush=True)
