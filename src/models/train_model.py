import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import json
import glob
from scipy import stats

class ModelTrainer:
    def __init__(self, data_path, symbol):
        """
        Initialize the model trainer
        Args:
            data_path (str): Path to the processed data file
            symbol (str): Symbol of the equity
        """
        self.data_path = data_path
        self.symbol = symbol
        self.models = {
            'random_forest': RandomForestRegressor(),
            'gradient_boosting': GradientBoostingRegressor(),
            'linear_regression': LinearRegression()
        }
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the data"""
        try:
            if self.data_path.endswith('.csv'):
                data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            else:
                data = pd.read_json(self.data_path)
            
            # Prepare features and target
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 
                       'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 
                       'BB_Upper', 'BB_Lower', 'Volume_SMA']
            
            X = data[features]
            y = data['Target']
            
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            print(f"Error loading data for {self.symbol}: {e}")
            return None, None, None, None
    
    def train_models(self, X_train, y_train):
        """Train multiple models and select the best one"""
        best_score = float('-inf')
        
        # Use TimeSeriesSplit for time series data
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.models.items():
            print(f"\nTraining {name} for {self.symbol}...")
            
            # Define expanded parameter grid for GridSearchCV
            if name == 'random_forest':
                param_grid = {
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', None]
                }
            elif name == 'gradient_boosting':
                param_grid = {
                    'n_estimators': [100, 200, 300, 500],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7, 9],
                    'min_samples_split': [2, 5, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:
                param_grid = {}
            
            # Perform grid search if parameters are specified
            if param_grid:
                grid_search = GridSearchCV(
                    model, param_grid, cv=tscv, scoring='r2', n_jobs=-1,
                    verbose=1
                )
                grid_search.fit(X_train, y_train)
                model = grid_search.best_estimator_
                score = grid_search.best_score_
                print(f"Best parameters for {name}: {grid_search.best_params_}")
            else:
                model.fit(X_train, y_train)
                score = model.score(X_train, y_train)
            
            print(f"{name} R² score for {self.symbol}: {score:.4f}")
            
            if score > best_score:
                best_score = score
                self.best_model = model
                self.best_model_name = name
    
    def calculate_prediction_metrics(self, y_true, y_pred):
        """Calculate comprehensive prediction metrics"""
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Percentage-based metrics
        percentage_errors = np.abs((y_true - y_pred) / y_true) * 100
        mean_percentage_error = np.mean(percentage_errors)
        median_percentage_error = np.median(percentage_errors)
        
        # Direction accuracy
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff))
        
        # Calculate confidence intervals
        confidence = 0.95
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std_error = np.std(y_true - y_pred)
        ci_lower = y_pred - z_score * std_error
        ci_upper = y_pred + z_score * std_error
        
        # Calculate prediction intervals
        prediction_interval = np.percentile(percentage_errors, 95)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mean_percentage_error': mean_percentage_error,
            'median_percentage_error': median_percentage_error,
            'direction_accuracy': direction_accuracy,
            'prediction_interval_95': prediction_interval,
            'confidence_interval': {
                'lower': ci_lower.tolist(),
                'upper': ci_upper.tolist()
            }
        }
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the best model on test data"""
        if self.best_model is None:
            return None
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'model_name': self.best_model_name,
            **self.calculate_prediction_metrics(y_test, y_pred)
        }
        
        # Feature importance for tree-based models
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = dict(zip(
                ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 
                 'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 
                 'BB_Upper', 'BB_Lower', 'Volume_SMA'],
                self.best_model.feature_importances_
            ))
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def save_model(self):
        """Save the best model and scaler"""
        if self.best_model is None:
            return
        
        os.makedirs('models', exist_ok=True)
        
        # Save model
        model_path = f'models/{self.symbol}_model.joblib'
        joblib.dump(self.best_model, model_path)
        
        # Save scaler
        scaler_path = f'models/{self.symbol}_scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")

def main():
    # Loop through all processed equity datasets
    data_files = glob.glob('data/*_processed.csv')
    for data_path in data_files:
        symbol = os.path.basename(data_path).split('_')[0]
        print(f"\n=== Training for {symbol} ===")
        trainer = ModelTrainer(data_path, symbol)
        X_train, X_test, y_train, y_test = trainer.load_data()
        
        if X_train is not None:
            trainer.train_models(X_train, y_train)
            metrics = trainer.evaluate_model(X_test, y_test)
            
            if metrics:
                print(f"\nModel Performance Metrics for {symbol}:")
                print(json.dumps(metrics, indent=2))
                
                # Save metrics
                with open(f'models/{symbol}_metrics.json', 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                trainer.save_model()
            else:
                print(f"No metrics for {symbol}.")
        else:
            print(f"No data for {symbol}.")

if __name__ == "__main__":
    main() 