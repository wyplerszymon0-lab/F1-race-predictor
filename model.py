import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from data import generate_season_data, DRIVERS, CIRCUIT_CHARACTERISTICS, TEAM_PERFORMANCE, DRIVER_SKILL


FEATURES = [
    "grid_position", "driver_skill", "team_performance",
    "is_street", "high_speed", "downforce_req",
    "is_wet", "is_mixed",
]


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURES].copy()
    y = df["won"]
    return X, y


def train_model(df: pd.DataFrame) -> tuple:
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    return model, acc, X_test, y_test


def predict_race(model, circuit: str, weather: str = "dry") -> pd.DataFrame:
    if circuit not in CIRCUIT_CHARACTERISTICS:
        available = list(CIRCUIT_CHARACTERISTICS.keys())
        raise ValueError(f"Unknown circuit. Available: {available}")

    circ = CIRCUIT_CHARACTERISTICS[circuit]
    rows = []

    grid_positions = list(range(1, len(DRIVERS) + 1))
    np.random.shuffle(grid_positions)

    for idx, (code, info) in enumerate(DRIVERS.items()):
        rows.append({
            "driver":           code,
            "name":             info["name"],
            "team":             info["team"],
            "grid_position":    grid_positions[idx],
            "driver_skill":     DRIVER_SKILL[code],
            "team_performance": TEAM_PERFORMANCE[info["team"]],
            "is_street":        circ["street"],
            "high_speed":       circ["high_speed"],
            "downforce_req":    circ["downforce"],
            "is_wet":           1 if weather == "wet"   else 0,
            "is_mixed":         1 if weather == "mixed" else 0,
        })

    df   = pd.DataFrame(rows)
    X    = df[FEATURES]
    probs = model.predict_proba(X)[:, 1]

    df["win_probability"] = probs
    df["win_probability"] = df["win_probability"] / df["win_probability"].sum()
    df["win_pct"]         = (df["win_probability"] * 100).round(1)

    return df.sort_values("win_probability", ascending=False).reset_index(drop=True)


def feature_importance(model) -> pd.DataFrame:
    return pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
