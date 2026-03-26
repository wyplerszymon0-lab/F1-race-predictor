# f1-race-predictor

Machine learning model that predicts Formula 1 race winners using
Gradient Boosting. Considers driver skill, team performance, circuit
characteristics and weather conditions.

## Features

- GradientBoostingClassifier trained on simulated historical data
- Predicts win probability for all 10 drivers
- Supports dry, wet and mixed weather conditions
- 15 circuits with unique characteristics
- Feature importance analysis

## Run
```bash
pip install -r requirements.txt

python main.py Britain
python main.py Monaco --weather wet
python main.py Italy --weather dry --importance
python main.py --circuits
```

## Example Output
```
Training model on historical data...
Model accuracy: 87.3%

==========================================================
  F1 RACE PREDICTOR — BRITAIN (DRY)
==========================================================
Pos  Driver                Team            Grid  Win %
----------------------------------------------------------
🥇   Max Verstappen        Red Bull        3     31.2%
🥈   Lewis Hamilton        Mercedes        1     18.4%
🥉   Lando Norris          McLaren         2     14.7%
  4. Charles Leclerc       Ferrari         5     11.3%
```

## Test
```bash
pytest tests/ -v
```

## Project Structure
```
f1-race-predictor/
├── data.py          # Dataset generation, driver/circuit data
├── model.py         # ML model training and prediction
├── main.py          # CLI entry point
├── requirements.txt
├── README.md
└── tests/
    └── test_model.py
```

## Author

**Szymon Wypler**
