import re
from dataclasses import dataclass
from itertools import accumulate
from numbers import Real
from pathlib import Path
from random import Random
from urllib.request import urlopen

import pandas as pd


DATA_URL = "https://datahub.io/core/speed-dating/_r/-/data/speed-dating.csv"
DATA_FILENAME = "speed-dating.csv"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = (
    "attractive_partner",
    "sincere_partner",
    "intelligence_partner",
    "funny_partner",
    "ambition_partner",
    "shared_interests_partner",
    "attractive_o",
    "sinsere_o",
    "intelligence_o",
    "funny_o",
    "ambitous_o",
    "shared_interests_o",
    "gender",
    "age",
    "age_o",
    "race",
    "race_o",
    "field",
    "attractive",
    "sincere",
    "intelligence",
    "funny",
    "ambition",
    "sports",
    "tvsports",
    "exercise",
    "dining",
    "museums",
    "art",
    "hiking",
    "gaming",
    "clubbing",
    "reading",
    "tv",
    "theater",
    "movies",
    "concerts",
    "music",
    "shopping",
    "yoga",
    "attractive_important",
    "sincere_important",
    "intellicence_important",
    "funny_important",
    "ambtition_important",
    "shared_interests_important",
    "pref_o_attractive",
    "pref_o_sincere",
    "pref_o_intelligence",
    "pref_o_funny",
    "pref_o_ambitious",
    "pref_o_shared_interests",
    "importance_same_race",
    "importance_same_religion",
    "expected_happy_with_sd_people",
    "d_age",
    "samerace",
    "interests_correlate",
    "met",
)

# Rule order matches EDA.ipynb because some field names fit several groups.
FIELD_GROUP_RULES = (
    (
        "business_finance_economics",
        r"business|mba|finance|finanace|econ|money|marketing|management",
    ),
    (
        "medicine_health",
        r"medicine|health|biomedical|microbiology|pathology|nursing|epidemiolog"
        r"|physiolog|nutrition",
    ),
    (
        "natural_sciences",
        r"biology|chemistry|physics|ecology|genetic|neuroscience|biochem|climate|environment"
        r"|marine|geophys|conservation|(?<!political )(?<!computer )science",
    ),
    (
        "stem_engineering_math",
        r"computer|engineering|math|statistic|operations research|financial math|applied maths",
    ),
    ("education", r"education|teaching|childhood|tesol|curriculum"),
    (
        "social_sciences_law",
        r"psychology|social work|sociology|political science|law|public administration"
        r"|international|anthropology|human rights|urban planning",
    ),
    (
        "humanities_arts",
        r"literature|german|polish|english|classics|philosophy|religion|history|art|film|journalism"
        r"|communications|creative writing|mfa|theater|theatre|writing|japanese|french"
        r"|american studies",
    ),
)


@dataclass
class PairSample:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def _load_data() -> pd.DataFrame:
    local_paths = (PROJECT_ROOT / DATA_FILENAME, PROJECT_ROOT / "notebooks" / DATA_FILENAME)
    source = next((path for path in local_paths if path.exists()), DATA_URL)
    try:
        if isinstance(source, Path):
            data = pd.read_csv(source)
        else:
            with urlopen(source, timeout=60) as response:
                data = pd.read_csv(response)
    except (OSError, ValueError) as error:
        raise ValueError(f"Could not read Speed Dating CSV from {source}: {error}") from error

    required_columns = [*FEATURE_COLUMNS, "wave", "match"]
    missing_columns = [column for column in required_columns if column not in data]
    if missing_columns:
        raise ValueError(f"CSV from {source} is missing columns: {', '.join(missing_columns)}")
    if data["wave"].isna().any() or data["wave"].nunique() < 3:
        raise ValueError(f"CSV from {source} needs a wave for each row and at least three waves")
    if not data["match"].isin([0, 1]).all():
        raise ValueError(f"CSV from {source} must have a binary (0 or 1) match value for each row")

    return data[required_columns].copy()


def _field_group(field: str | float) -> str:
    if pd.isna(field):
        return "unknown"

    normalized = field.lower()
    for group, pattern in FIELD_GROUP_RULES:
        if re.search(pattern, normalized):
            return group
    return "other"


def _prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    rating_columns = ["reading", "gaming", "funny_o", "attractive_o"]
    data[rating_columns] = data[rating_columns].clip(upper=10)
    data["met"] = data["met"].mask(data["met"] > 1)
    data["d_age"] = data["d_age"].mask(data[["age", "age_o"]].isna().any(axis=1))
    for column in ("field", "race", "race_o"):
        data[column] = data[column].astype("string").str.strip("'")
    data["field_group"] = data["field"].map(_field_group)
    return data


def _split_waves(
    wave_counts: pd.Series, seed: int, proportions: tuple[float, float, float]
) -> tuple[list[int], list[int], list[int]]:
    counts = wave_counts.sort_index().to_dict()
    waves = list(counts)
    total_rows = sum(counts.values())
    train_target, valid_target, test_target = (total_rows * size for size in proportions)
    rng = Random(seed)
    best_error = float("inf")
    best_split: tuple[list[int], list[int], list[int]] = ([], [], [])

    # Use only wave sizes to select the split, keeping the target out of this step.
    for _ in range(512):
        order = rng.sample(waves, len(waves))
        cumulative_rows = list(accumulate(counts[wave] for wave in order))
        for train_end in range(1, len(waves) - 1):
            train_rows = cumulative_rows[train_end - 1]
            train_error = (train_rows - train_target) ** 2
            for valid_end in range(train_end + 1, len(waves)):
                valid_rows = cumulative_rows[valid_end - 1] - train_rows
                test_rows = total_rows - cumulative_rows[valid_end - 1]
                error = (
                    train_error + (valid_rows - valid_target) ** 2 + (test_rows - test_target) ** 2
                )
                if error < best_error:
                    best_error = error
                    best_split = (order[:train_end], order[train_end:valid_end], order[valid_end:])

    return best_split


def get_sample(seed: int, *, train_size: float = 0.70, valid_size: float = 0.15) -> PairSample:
    """Return train, validation, and test frames with `match` as the target.

    Sizes are target fractions of the rows. Test gets the remaining fraction.
    Actual sizes can differ because each wave stays together.
    """
    valid_fractions = all(
        isinstance(size, Real) and 0 < size < 1 for size in (train_size, valid_size)
    )
    if not valid_fractions or train_size + valid_size >= 1:
        raise ValueError("train_size and valid_size must be positive fractions with a sum below 1")

    test_size = 1 - (train_size + valid_size)
    data = _prepare_features(_load_data())
    wave_counts = data["wave"].value_counts()
    split_waves = _split_waves(wave_counts, seed, (train_size, valid_size, test_size))
    train, valid, test = (
        data.loc[data["wave"].isin(waves)].drop(columns="wave").reset_index(drop=True)
        for waves in split_waves
    )
    return PairSample(train=train, valid=valid, test=test)
