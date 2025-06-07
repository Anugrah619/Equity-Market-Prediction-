# Equity Market Prediction Model

This project implements a machine learning-based equity market prediction system with a web interface for visualization and predictions.

## Project Structure
```
equity_prediction/
├── data/                  # Data storage directory
├── models/               # Trained model storage
├── src/
│   ├── data/            # Data processing scripts
│   ├── features/        # Feature engineering scripts
│   ├── models/          # Model training scripts
│   └── web/             # Web application files
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Data Collection and Preprocessing:
```bash
python src/data/data_collector.py
```

2. Model Training:
```bash
python src/models/train_model.py
```

3. Run Web Application:
```bash
python src/web/app.py
```

## Features

- Historical equity data collection and preprocessing
- Feature engineering and selection
- Machine learning model training and evaluation
- Interactive web interface with:
  - Real-time price predictions
  - Performance metrics visualization
  - Historical data analysis
  - Model performance dashboard

## Model Performance Metrics

The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared Score
- Feature Importance Analysis

## License

MIT License 