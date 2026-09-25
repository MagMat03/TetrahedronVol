import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

DATA_TRAINING = Path("data/tsgr_dataset/training")
DATA_TESTING = Path("data/tsgr_dataset/testing")
MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)


def extract_world_landmarks_pure_mp(dataset_dir: Path):
    world_samples = []

    with vision.HandLandmarker.create_from_options(options) as detector:

        image_paths = sorted(list(dataset_dir.glob("**/*.jpg")))
        print(f"Znaleziono {len(image_paths)} zdjęć w bazie.")

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = detector.detect(mp_image)
            if result.hand_world_landmarks:
                world_lm = result.hand_world_landmarks[0]
                landmarks_matrix = np.array([[lm.x, lm.y, lm.z] for lm in world_lm])
                world_samples.append(landmarks_matrix)
            else:
                world_samples.append(np.zeros((21, 3)))

    return np.array(world_samples)


#test
X_world_3d_train = extract_world_landmarks_pure_mp(DATA_TRAINING)
X_world_3d_test = extract_world_landmarks_pure_mp(DATA_TESTING)
print(f"Kształt macierzy końcowej treningowej: {X_world_3d_train.shape}")
print(f"Kształt macierzy końcowej testowej: {X_world_3d_test.shape}")