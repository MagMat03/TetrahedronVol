import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
TRAINING_LANDMARKS_FILE = RESULTS_DIR / "training_landmarks.json"
TESTING_LANDMARKS_ROOT = RESULTS_DIR / "testing_landmarks"
ANNOTATIONS_PATH = Path("data/tsgr_dataset/annotations.csv")

OUTPUT_TRAINING_FEATURES = RESULTS_DIR / "training_features.csv"
OUTPUT_TESTING_FEATURES = RESULTS_DIR / "testing_features.csv"

# Punkty bazowe plaszczyzny odniesienia dloni (P0, P5, P17 z dokumentu)
BASE_INDICES = (0, 5, 17)
# 18 pozostalych punktow, dla ktorych liczymy objetosci - rosnaca kolejnosc indeksow
FEATURE_INDICES = [i for i in range(21) if i not in BASE_INDICES]
FEATURE_NAMES = [f"V{idx}_norm" for idx in FEATURE_INDICES]


# ---------------------------------------------------------------------------
# 1) Obliczanie cech - metoda objetosci czworoscianu
# ---------------------------------------------------------------------------

def compute_tetrahedron_features(world_landmarks: list[list[float]]) -> list[float]:
    """Oblicza 18-elementowy wektor znormalizowanych, zorientowanych objetosci
    czworoscianow V_i^norm, zgodnie z dokumentem "Metoda objetosci czworoscianu
    w zadaniu klasyfikacji gestow statycznych".

    Plaszczyzna odniesienia: trojkat (P0, P5, P17).
    Skala liniowa dloni: s = ||P17 - P5||.
    Dla kazdego P_i (i spoza {0,5,17}):
        S_i = [(P5-P0) x (P17-P0)] . (P_i-P0)
        V_i = S_i / 6
        V_i^norm = V_i / s^3

    Parametry
    ---------
    world_landmarks: lista 21 punktow [x, y, z] (world landmarks z MediaPipe).

    Zwraca
    ------
    Liste 18 wartosci float, w kolejnosci rosnacych indeksow punktow
    (pomijajac P0, P5, P17).
    """
    points = np.asarray(world_landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Oczekiwano 21 punktow 3D, otrzymano ksztalt {points.shape}")

    p0 = points[0]
    p5 = points[5]
    p17 = points[17]

    scale = np.linalg.norm(p17 - p5)
    if scale <= 1e-9:
        # Brak wiarygodnej detekcji dloni (np. wektor zerowy) - zwracamy zera,
        # zeby nie dzielic przez zero i nie wywalac calego przetwarzania.
        return [0.0] * len(FEATURE_INDICES)

    v_base1 = p5 - p0
    v_base2 = p17 - p0
    cross = np.cross(v_base1, v_base2)

    features: list[float] = []
    for idx in FEATURE_INDICES:
        pi = points[idx]
        signed_volume = (1.0 / 6.0) * np.dot(cross, pi - p0)
        normalized = signed_volume / (scale ** 3)
        features.append(float(normalized))

    return features


# ---------------------------------------------------------------------------
# 2) Wczytanie annotations.csv
# ---------------------------------------------------------------------------

AnnotationKey = tuple[str, str, str, str]  # (subject, background, gesture, take_id)


def load_annotations(annotations_path: Path) -> dict[AnnotationKey, dict[str, int]]:
    """Wczytuje annotations.csv i zwraca slownik:
    (subject, background, gesture, take_id) -> {frame_count, gesture_start_frame, gesture_end_frame}
    """
    lookup: dict[AnnotationKey, dict[str, int]] = {}
    with annotations_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["public_subject_id"], row["background"], row["gesture"], row["take_id"])
            lookup[key] = {
                "frame_count": int(row["frame_count"]),
                "gesture_start_frame": int(row["gesture_start_frame"]),
                "gesture_end_frame": int(row["gesture_end_frame"]),
            }
    print(f"Wczytano {len(lookup)} adnotacji z {annotations_path}")
    return lookup


# ---------------------------------------------------------------------------
# 3a) Cechy dla danych treningowych (pojedyncze zdjecia, bez zakresu klatek)
# ---------------------------------------------------------------------------

def build_training_feature_table(training_landmarks_file: Path) -> list[dict[str, Any]]:
    """Wczytuje training_landmarks.json (wynik process_training_images) i dolicza
    do kazdej probki 18-elementowy wektor cech."""
    with training_landmarks_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    table: list[dict[str, Any]] = []
    for record in records:
        features = compute_tetrahedron_features(record["world_landmarks"])
        row: dict[str, Any] = {
            "subject": record["subject"],
            "background": record["background"],
            "gesture": record["gesture"],
            "file": record["file"],
        }
        row.update(zip(FEATURE_NAMES, features))
        table.append(row)

    print(f"Policzono cechy dla {len(table)} probek treningowych")
    return table


# ---------------------------------------------------------------------------
# 3b) Cechy dla danych testowych + polaczenie z annotations.csv
# ---------------------------------------------------------------------------

def build_testing_feature_table(
    testing_landmarks_root: Path,
    annotations_path: Path,
    only_gesture_frames: bool = True,
) -> list[dict[str, Any]]:
    """Przechodzi po wszystkich testing_landmarks/**/landmarks.json (wynik
    process_testing_takes), laczy kazdy take z annotations.csv po kluczu
    (subject, background, gesture, take_id) i liczy cechy dla kazdej klatki.

    only_gesture_frames=True  -> zwraca tylko klatki z zakresu
        [gesture_start_frame, gesture_end_frame] (gest faktycznie "trzymany").
    only_gesture_frames=False -> zwraca wszystkie klatki, z dodatkowa kolumna
        is_gesture_frame (True/False).
    """
    annotations = load_annotations(annotations_path)

    take_files = sorted(testing_landmarks_root.rglob("landmarks.json"))
    print(f"Znaleziono {len(take_files)} plikow landmarks.json do polaczenia z adnotacjami")

    table: list[dict[str, Any]] = []
    missing_annotations = 0

    for take_file in take_files:
        with take_file.open("r", encoding="utf-8") as f:
            take_data = json.load(f)

        subject = take_data["subject"]
        background = take_data["background"]
        gesture = take_data["gesture"]
        take_id = take_data["take_id"]

        key = (subject, background, gesture, take_id)
        annotation = annotations.get(key)
        if annotation is None:
            missing_annotations += 1
            print(f"[UWAGA] Brak adnotacji dla {key} - pomijam ten take")
            continue

        start_frame = annotation["gesture_start_frame"]
        end_frame = annotation["gesture_end_frame"]

        for frame in take_data["frames"]:
            frame_idx = frame["frame_index"]
            is_gesture_frame = start_frame <= frame_idx <= end_frame

            if only_gesture_frames and not is_gesture_frame:
                continue

            features = compute_tetrahedron_features(frame["world_landmarks"])
            row: dict[str, Any] = {
                "subject": subject,
                "background": background,
                "gesture": gesture,
                "take_id": take_id,
                "frame_index": frame_idx,
                "file": frame["file"],
                "is_gesture_frame": is_gesture_frame,
            }
            row.update(zip(FEATURE_NAMES, features))
            table.append(row)

    if missing_annotations:
        print(f"[UWAGA] Lacznie {missing_annotations} take'ow bez dopasowanej adnotacji")

    print(f"Policzono cechy dla {len(table)} klatek testowych")
    return table


# ---------------------------------------------------------------------------
# Zapis do CSV
# ---------------------------------------------------------------------------

def save_table_to_csv(table: list[dict[str, Any]], output_file: Path) -> None:
    if not table:
        print(f"Brak danych do zapisania w {output_file}")
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(table[0].keys())
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    print(f"Zapisano {len(table)} wierszy do {output_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # 1) cechy dla danych treningowych
    training_table = build_training_feature_table(TRAINING_LANDMARKS_FILE)
    save_table_to_csv(training_table, OUTPUT_TRAINING_FEATURES)

    # 2) cechy dla danych testowych, polaczone z annotations.csv
    testing_table = build_testing_feature_table(
        testing_landmarks_root=TESTING_LANDMARKS_ROOT,
        annotations_path=ANNOTATIONS_PATH,
        only_gesture_frames=True,
    )
    save_table_to_csv(testing_table, OUTPUT_TESTING_FEATURES)