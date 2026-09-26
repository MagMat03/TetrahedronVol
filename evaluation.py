"""Generuje foldy S1 (all-in-domain), S2 (LOBO), S3 (LOSO) i S4 (LOSO+background)
na podstawie juz policzonych plikow cech: training_features.csv i testing_features.csv.

Logika podzialu train/test jest 1:1 taka sama jak w oryginalnym folds.py z TSGR-F.
Roznica: tam trzeba bylo dodatkowo lokalizowac na dysku przetworzone "runy" cech dla
kazdej komorki (gest, subject, background) - tutaj cechy sa juz policzone w CSV,
wiec kazdy fold to po prostu filtrowanie wierszy po kolumnach subject/background
i zapisanie odpowiedniego podzbioru do osobnych plikow CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCENARIOS = ("S1_ALL_IN_DOMAIN", "S2_LOBO", "S3_LOSO", "S4_LOSO_BACKGROUND")


@dataclass(frozen=True, slots=True)
class FoldDefinition:
    scenario: str
    fold_id: str
    held_out_subject: str | None = None
    held_out_background: str | None = None


# ---------------------------------------------------------------------------
# I/O pomocnicze
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Definicje foldow (identyczne jak w oryginale)
# ---------------------------------------------------------------------------

def build_fold_definitions(subjects: Iterable[str], backgrounds: Iterable[str]) -> list[FoldDefinition]:
    subjects = sorted(set(subjects))
    backgrounds = sorted(set(backgrounds))
    folds = [FoldDefinition("S1_ALL_IN_DOMAIN", "all")]
    folds.extend(
        FoldDefinition("S2_LOBO", f"background_{background}", held_out_background=background)
        for background in backgrounds
    )
    folds.extend(
        FoldDefinition("S3_LOSO", f"subject_{subject}", held_out_subject=subject)
        for subject in subjects
    )
    folds.extend(
        FoldDefinition(
            "S4_LOSO_BACKGROUND",
            f"subject_{subject}__background_{background}",
            held_out_subject=subject,
            held_out_background=background,
        )
        for subject in subjects
        for background in backgrounds
    )
    return folds


def _training_selected(row: dict[str, str], fold: FoldDefinition) -> bool:
    if fold.scenario == "S1_ALL_IN_DOMAIN":
        return True
    if fold.scenario == "S2_LOBO":
        return row["background"] != fold.held_out_background
    if fold.scenario == "S3_LOSO":
        return row["subject"] != fold.held_out_subject
    if fold.scenario == "S4_LOSO_BACKGROUND":
        return row["subject"] != fold.held_out_subject and row["background"] != fold.held_out_background
    raise ValueError(f"Unsupported scenario: {fold.scenario}")


def _test_selected(row: dict[str, str], fold: FoldDefinition) -> bool:
    if fold.scenario == "S1_ALL_IN_DOMAIN":
        return True
    if fold.scenario == "S2_LOBO":
        return row["background"] == fold.held_out_background
    if fold.scenario == "S3_LOSO":
        return row["subject"] == fold.held_out_subject
    if fold.scenario == "S4_LOSO_BACKGROUND":
        return row["subject"] == fold.held_out_subject and row["background"] == fold.held_out_background
    raise ValueError(f"Unsupported scenario: {fold.scenario}")


# ---------------------------------------------------------------------------
# Glowna funkcja
# ---------------------------------------------------------------------------

def generate_experiment_folds(
    training_features_path: str | Path,
    testing_features_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    scenarios: Iterable[str] | None = None,
    progress: bool = True,
) -> Path:
    """Generuje foldy S1/S2/S3/S4 na podstawie training_features.csv i
    testing_features.csv. Dla kazdego aktywnego folda zapisuje osobne pliki
    training_features.csv / testing_features.csv (podzbior wierszy) oraz fold.json
    z podsumowaniem. Na koniec zapisuje zbiorczy experiment_plan.json.
    """
    training_path = Path(training_features_path)
    testing_path = Path(testing_features_path)

    training = _read_csv(training_path)
    testing = _read_csv(testing_path)
    if not training or not testing:
        raise ValueError(
            f"Brak danych: training={len(training)} wierszy, testing={len(testing)} wierszy. "
            f"Sprawdz sciezki {training_path} / {testing_path}."
        )

    training_fields = list(training[0].keys())
    testing_fields = list(testing[0].keys())

    subjects = sorted({row["subject"] for row in training} | {row["subject"] for row in testing})
    backgrounds = sorted({row["background"] for row in training} | {row["background"] for row in testing})

    selected_scenarios = set(scenarios or SCENARIOS)
    unknown = selected_scenarios - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Nieznane scenariusze: {sorted(unknown)}")

    plan_dir = (
        Path(output_dir)
        if output_dir
        else Path("reports") / "experiments" / datetime.now().strftime("experiment_plan_%Y%m%d_%H%M%S")
    )
    plan_dir.mkdir(parents=True, exist_ok=False)

    folds = [
        fold
        for fold in build_fold_definitions(subjects, backgrounds)
        if fold.scenario in selected_scenarios
    ]

    fold_summaries: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        fold_dir = plan_dir / fold.scenario / fold.fold_id
        fold_dir.mkdir(parents=True, exist_ok=True)

        selected_training = [row for row in training if _training_selected(row, fold)]
        selected_testing = [row for row in testing if _test_selected(row, fold)]

        if not selected_training and not selected_testing:
            fold_status = "deferred_no_training_or_test_data"
        elif not selected_training:
            fold_status = "deferred_no_training_data"
        elif not selected_testing:
            fold_status = "deferred_no_test_data"
        else:
            fold_status = "active"

        if fold_status == "active":
            _write_csv(fold_dir / "training_features.csv", selected_training, training_fields)
            _write_csv(fold_dir / "testing_features.csv", selected_testing, testing_fields)

        train_gestures = sorted({row["gesture"] for row in selected_training})
        test_gestures = sorted({row["gesture"] for row in selected_testing})

        payload: dict[str, Any] = {
            **asdict(fold),
            "fold_status": fold_status,
            "deferred": fold_status != "active",
            "training_row_count": len(selected_training),
            "testing_row_count": len(selected_testing),
            "training_subjects": sorted({row["subject"] for row in selected_training}),
            "training_backgrounds": sorted({row["background"] for row in selected_training}),
            "training_gestures": train_gestures,
            "test_subjects": sorted({row["subject"] for row in selected_testing}),
            "test_backgrounds": sorted({row["background"] for row in selected_testing}),
            "test_gestures": test_gestures,
            "ready_for_model_building": fold_status == "active",
            "ready_for_evaluation": fold_status == "active" and train_gestures == test_gestures,
            "leakage_policy": (
                "Model (progi decyzyjne, wagi itp.) nalezy trenowac wylacznie na "
                "training_features.csv z tego folda - nigdy na danych z folderu testing."
            ),
        }
        with (fold_dir / "fold.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        fold_summaries.append(payload)

        if progress:
            print(
                f"[{fold_number}/{len(folds)}] {fold.scenario}/{fold.fold_id}: {fold_status} "
                f"train={len(selected_training)} test={len(selected_testing)}",
                flush=True,
            )

    with (plan_dir / "experiment_plan.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "tsgr_custom_experiment_plan_v1",
                "subjects": subjects,
                "backgrounds": backgrounds,
                "scenarios": sorted(selected_scenarios),
                "fold_count": len(fold_summaries),
                "active_fold_count": sum(row["fold_status"] == "active" for row in fold_summaries),
                "deferred_fold_count": sum(row["fold_status"] != "active" for row in fold_summaries),
                "folds": fold_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return plan_dir


if __name__ == "__main__":
    plan_dir = generate_experiment_folds(
        training_features_path="training_features.csv",
        testing_features_path="testing_features.csv",
    )
    print(f"\nZapisano plan eksperymentow w: {plan_dir}")