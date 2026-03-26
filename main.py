import argparse
import sys
from data import generate_season_data, CIRCUIT_CHARACTERISTICS
from model import train_model, predict_race, feature_importance


def print_prediction(result, circuit, weather):
    print(f"\n{'='*58}")
    print(f"  F1 RACE PREDICTOR — {circuit.upper()} ({weather.upper()})")
    print(f"{'='*58}")
    print(f"{'Pos':<5}{'Driver':<22}{'Team':<16}{'Grid':<6}{'Win %'}")
    print(f"{'-'*58}")

    for i, row in result.iterrows():
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        print(f"{medal:<5}{row['name']:<22}{row['team']:<16}{int(row['grid_position']):<6}{row['win_pct']}%")

    print(f"{'='*58}\n")


def main():
    parser = argparse.ArgumentParser(description="F1 Race Predictor")
    parser.add_argument("circuit", nargs="?", default="Britain",    help="Circuit name")
    parser.add_argument("--weather", default="dry",                 choices=["dry", "wet", "mixed"])
    parser.add_argument("--circuits", action="store_true",          help="List available circuits")
    parser.add_argument("--importance", action="store_true",        help="Show feature importance")
    args = parser.parse_args()

    if args.circuits:
        print("\nAvailable circuits:")
        for c in CIRCUIT_CHARACTERISTICS:
            print(f"  - {c}")
        return

    print("Training model on historical data...")
    df            = generate_season_data(n_races=300)
    model, acc, _, _ = train_model(df)
    print(f"Model accuracy: {acc:.1%}\n")

    if args.importance:
        fi = feature_importance(model)
        print("Feature Importance:")
        for _, row in fi.iterrows():
            bar = "█" * int(row["importance"] * 50)
            print(f"  {row['feature']:<20} {bar} {row['importance']:.3f}")
        print()

    try:
        result = predict_race(model, args.circuit, args.weather)
        print_prediction(result, args.circuit, args.weather)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
