import pandas as pd
import numpy as np

DRIVERS = {
    "VER": {"name": "Max Verstappen",    "team": "Red Bull",   "number": 1},
    "PER": {"name": "Sergio Perez",      "team": "Red Bull",   "number": 11},
    "LEC": {"name": "Charles Leclerc",   "team": "Ferrari",    "number": 16},
    "SAI": {"name": "Carlos Sainz",      "team": "Ferrari",    "number": 55},
    "HAM": {"name": "Lewis Hamilton",    "team": "Mercedes",   "number": 44},
    "RUS": {"name": "George Russell",    "team": "Mercedes",   "number": 63},
    "NOR": {"name": "Lando Norris",      "team": "McLaren",    "number": 4},
    "PIA": {"name": "Oscar Piastri",     "team": "McLaren",    "number": 81},
    "ALO": {"name": "Fernando Alonso",   "team": "Aston Martin","number": 14},
    "STR": {"name": "Lance Stroll",      "team": "Aston Martin","number": 18},
}

CIRCUITS = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

TEAM_PERFORMANCE = {
    "Red Bull":    0.95,
    "Ferrari":     0.88,
    "McLaren":     0.87,
    "Mercedes":    0.85,
    "Aston Martin":0.78,
}

DRIVER_SKILL = {
    "VER": 0.98, "PER": 0.82,
    "LEC": 0.91, "SAI": 0.88,
    "HAM": 0.93, "RUS": 0.87,
    "NOR": 0.90, "PIA": 0.85,
    "ALO": 0.89, "STR": 0.75,
}

CIRCUIT_CHARACTERISTICS = {
    "Bahrain":        {"street": 0, "high_speed": 0.5, "downforce": 0.6},
    "Saudi Arabia":   {"street": 0.7, "high_speed": 0.9, "downforce": 0.5},
    "Australia":      {"street": 0.3, "high_speed": 0.6, "downforce": 0.6},
    "Japan":          {"street": 0, "high_speed": 0.8, "downforce": 0.9},
    "China":          {"street": 0, "high_speed": 0.6, "downforce": 0.7},
    "Miami":          {"street": 0.5, "high_speed": 0.7, "downforce": 0.6},
    "Monaco":         {"street": 1.0, "high_speed": 0.1, "downforce": 0.9},
    "Canada":         {"street": 0.4, "high_speed": 0.7, "downforce": 0.5},
    "Spain":          {"street": 0, "high_speed": 0.6, "downforce": 0.8},
    "Britain":        {"street": 0, "high_speed": 0.9, "downforce": 0.7},
    "Hungary":        {"street": 0, "high_speed": 0.3, "downforce": 0.9},
    "Belgium":        {"street": 0, "high_speed": 1.0, "downforce": 0.6},
    "Italy":          {"street": 0, "high_speed": 1.0, "downforce": 0.3},
    "Singapore":      {"street": 1.0, "high_speed": 0.2, "downforce": 0.9},
    "Abu Dhabi":      {"street": 0, "high_speed": 0.7, "downforce": 0.6},
}


def generate_season_data(n_races: int = 200, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    rows = []

    for race_id in range(n_races):
        circuit = np.random.choice(list(CIRCUIT_CHARACTERISTICS.keys()))
        circ    = CIRCUIT_CHARACTERISTICS[circuit]
        weather = np.random.choice(["dry", "wet", "mixed"], p=[0.7, 0.15, 0.15])

        qualifying_positions = list(range(1, len(DRIVERS) + 1))
        np.random.shuffle(qualifying_positions)

        for idx, (code, info) in enumerate(DRIVERS.items()):
            team_perf   = TEAM_PERFORMANCE[info["team"]]
            driver_skill = DRIVER_SKILL[code]

            wet_bonus = 0
            if weather in ("wet", "mixed"):
                wet_bonus = np.random.uniform(-0.05, 0.1) if code in ("HAM", "VER", "ALO") else np.random.uniform(-0.1, 0.05)

            street_bonus = circ["street"] * (0.05 if code in ("LEC", "ALO", "HAM") else -0.02)
            grid_pos     = qualifying_positions[idx]
            noise        = np.random.normal(0, 0.03)

            score = (
                driver_skill * 0.40 +
                team_perf    * 0.35 +
                (1 - grid_pos / len(DRIVERS)) * 0.15 +
                wet_bonus    * 0.05 +
                street_bonus * 0.05 +
                noise
            )

            finish_pos = None
            rows.append({
                "race_id":        race_id,
                "circuit":        circuit,
                "weather":        weather,
                "driver":         code,
                "team":           info["team"],
                "grid_position":  grid_pos,
                "driver_skill":   driver_skill,
                "team_performance": team_perf,
                "is_street":      circ["street"],
                "high_speed":     circ["high_speed"],
                "downforce_req":  circ["downforce"],
                "is_wet":         1 if weather == "wet"   else 0,
                "is_mixed":       1 if weather == "mixed" else 0,
                "score":          score,
            })

    df = pd.DataFrame(rows)

    for race_id in df["race_id"].unique():
        mask  = df["race_id"] == race_id
        ranks = df.loc[mask, "score"].rank(ascending=False).astype(int)
        df.loc[mask, "finish_position"] = ranks

    df["won"] = (df["finish_position"] == 1).astype(int)
    df["podium"] = (df["finish_position"] <= 3).astype(int)

    return df.drop(columns=["score"])
