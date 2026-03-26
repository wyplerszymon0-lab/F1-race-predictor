import pytest
import pandas as pd
import numpy as np
from data import generate_season_data, DRIVERS, CIRCUIT_CHARACTERISTICS
from model import train_model, predict_race, prepare_data, feature_importance, FEATURES


@pytest.fixture(scope="module")
def dataset():
    return generate_season_data(n_races=100, seed=42)


@pytest.fixture(scope="module")
def trained_model(dataset):
    model, acc, _, _ = train_model(dataset)
    return model, acc


def test_dataset_has_correct_columns(dataset):
    required = ["race_id", "driver", "team", "grid_position",
                "finish_position", "won", "podium", "weather", "circuit"]
    for col in required:
        assert col in dataset.columns


def test_dataset_has_all_drivers(dataset):
    for code in DRIVERS:
        assert code in dataset["driver"].values


def test_each_race_has_one_winner(dataset):
    winners = dataset.groupby("race_id")["won"].sum()
    assert (winners == 1).all()


def test_finish_positions_are_unique_per_race(dataset):
    for _, group in dataset.groupby("race_id"):
        positions = group["finish_position"].tolist()
        assert len(positions) == len(set(positions))


def test_prepare_data_returns_correct_features(dataset):
    X, y = prepare_data(dataset)
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)


def test_model_trains_successfully(trained_model):
    model, acc = trained_model
    assert model is not None
    assert 0.0 < acc <= 1.0


def test_predict_race_returns_all_drivers(trained_model):
    model, _ = trained_model
    result   = predict_race(model, "Britain", "dry")
    assert len(result) == len(DRIVERS)


def test_predict_race_probabilities_sum_to_one(trained_model):
    model, _ = trained_model
    result   = predict_race(model, "Britain", "dry")
    total    = result["win_probability"].sum()
    assert abs(total - 1.0) < 0.001


def test_predict_race_sorted_by_probability(trained_model):
    model, _ = trained_model
    result   = predict_race(model, "Britain", "dry")
    probs    = result["win_probability"].tolist()
    assert probs == sorted(probs, reverse=True)


def test_predict_race_wet_differs_from_dry(trained_model):
    model, _ = trained_model
    dry = predict_race(model, "Britain", "dry")
    wet = predict_race(model, "Britain", "wet")
    assert not dry["win_probability"].equals(wet["win_probability"])


def test_predict_race_unknown_circuit_raises(trained_model):
    model, _ = trained_model
    with pytest.raises(ValueError, match="Unknown circuit"):
        predict_race(model, "Nonexistent Circuit")


def test_feature_importance_has_all_features(trained_model):
    model, _ = trained_model
    fi       = feature_importance(model)
    assert set(fi["feature"].tolist()) == set(FEATURES)


def test_feature_importance_sums_to_one(trained_model):
    model, _ = trained_model
    fi       = feature_importance(model)
    assert abs(fi["importance"].sum() - 1.0) < 0.001
```

---

**`requirements.txt`**
```
scikit-learn==1.4.0
pandas==2.2.0
numpy==1.26.0
pytest==8.2.0
