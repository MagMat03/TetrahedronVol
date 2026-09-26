import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
from typing import Any

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_ROOT = Path("data/tsgr_dataset")
MODEL_PATH = "hand_landmarker.task"


def _create_landmarker(mode: str) -> vision.HandLandmarker:
    running_mode = (
        vision.RunningMode.IMAGE if mode == "image" else vision.RunningMode.VIDEO
    )
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)


def extract_world_landmarks_from_image(result: Any) -> list[list[float]]:
    if not result.hand_world_landmarks:
        return np.zeros((21, 3)).tolist()
    world_lm = result.hand_world_landmarks[0]
    return [[lm.x, lm.y, lm.z] for lm in world_lm]


def process_training_images(training_dir: Path, output_file: Path) -> None:
    image_paths = sorted(list(training_dir.rglob("images/*.jpg")))
    total_images = len(image_paths)
    print(f"Rozpoczynam przetwarzanie: znaleziono {total_images} zdjęć treningowych w {training_dir}")

    results = []

    with _create_landmarker(mode="image") as detector:
        for idx, img_path in enumerate(image_paths, start=1):
            parts = img_path.parts
            subject, background, gesture = parts[-5], parts[-4], parts[-3]

            image = cv2.imread(str(img_path))
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                detection_result = detector.detect(mp_image)
                landmarks = extract_world_landmarks_from_image(detection_result)

                results.append({
                    "subject": subject,
                    "background": background,
                    "gesture": gesture,
                    "file": img_path.name,
                    "world_landmarks": landmarks
                })

            # Logowanie postępu co 100 zdjęć
            if idx % 100 == 0:
                remaining = total_images - idx
                print(f"[Trening] Przetworzono {idx}/{total_images}. Pozostało do końca: {remaining}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Zakończono. Zapisano dane treningowe do {output_file}\n")


def process_testing_takes(testing_dir: Path, output_root: Path) -> None:
    take_dirs = sorted([d for d in testing_dir.rglob("take_*") if d.is_dir()])

    takes_with_frames = []
    total_frames = 0
    for take_dir in take_dirs:
        frame_paths = sorted(list(take_dir.rglob("frames/*.jpg")))
        if frame_paths:
            takes_with_frames.append((take_dir, frame_paths))
            total_frames += len(frame_paths)

    print(
        f"Rozpoczynam przetwarzanie: znaleziono {total_frames} klatek testowych w {len(takes_with_frames)} nagraniach (takes)")

    processed_frames = 0
    global_timestamp_ms = 0  # <--- DODANO: Globalny zegar

    with _create_landmarker(mode="video") as detector:
        for take_dir, frame_paths in takes_with_frames:
            parts = take_dir.parts
            subject, background, gesture, take_id = parts[-4], parts[-3], parts[-2], parts[-1]
            take_results = []

            for frame_idx, frame_path in enumerate(frame_paths):
                image = cv2.imread(str(frame_path))
                if image is not None:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                    # Używamy globalnego zegara i od razu dodajemy mu 33ms
                    detection_result = detector.detect_for_video(mp_image, global_timestamp_ms)
                    global_timestamp_ms += 33  # <--- DODANO: Zwiększamy czas dla WSZYSTKICH klatek

                    landmarks = extract_world_landmarks_from_image(detection_result)

                    take_results.append({
                        "frame_index": frame_idx,
                        "file": frame_path.name,
                        "world_landmarks": landmarks
                    })

                processed_frames += 1

                if processed_frames % 100 == 0:
                    remaining = total_frames - processed_frames
                    print(
                        f"[Testy wideo] Przetworzono {processed_frames}/{total_frames} klatek. Pozostało do końca: {remaining}")

            take_output_dir = output_root / subject / background / gesture / take_id
            take_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = take_output_dir / "landmarks.json"

            with output_file.open("w", encoding="utf-8") as f:
                json.dump({
                    "subject": subject,
                    "background": background,
                    "gesture": gesture,
                    "take_id": take_id,
                    "frames": take_results
                }, f, indent=2)

    print(f"Zakończono. Zapisano wszystkie wyniki testowe w folderze {output_root}")

if __name__ == "__main__":
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    process_training_images(
        training_dir=DATASET_ROOT / "training",
        output_file=out_dir / "training_landmarks.json"
    )

    process_testing_takes(
        testing_dir=DATASET_ROOT / "testing",
        output_root=out_dir / "testing_landmarks"
    )