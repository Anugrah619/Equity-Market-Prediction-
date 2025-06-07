import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class DataCollector:
    def __init__(self, symbol='^GSPC', start_date=None, end_date=None):
        """
        Initialize the data collector with stock symbol and date range
        Args:
            symbol (str): Stock symbol (default: S&P 500)
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.symbol = symbol
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
    def fetch_data(self):
        """Fetch historical data from Yahoo Finance"""
        try:
            data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            print("Fetched columns:", data.columns)
            # Flatten columns if MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(col).strip() for col in data.columns.values]
            # Rename columns back to original names
            data = data.rename(columns={
                f'Close_{self.symbol}': 'Close',
                f'High_{self.symbol}': 'High',
                f'Low_{self.symbol}': 'Low',
                f'Open_{self.symbol}': 'Open',
                f'Volume_{self.symbol}': 'Volume'
            })
            return data
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {e}")
            return None

    def preprocess_data(self, data):
        """Preprocess the raw data"""
        if data is None or data.empty:
            return None

        # Calculate technical indicators
        df = data.copy()
        
        # Ensure 'Close' is a Series, not a DataFrame
        if isinstance(df['Close'], pd.DataFrame):
            df['Close'] = df['Close'].iloc[:, 0]
        # If columns are multi-index, flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        # Debug print for columns if error persists
        # print('Columns:', df.columns)
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Target variable (next day's return)
        df['Target'] = df['Close'].pct_change().shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    def save_data(self, data, format='csv'):
        """Save the processed data"""
        if data is None:
            return
        
        os.makedirs('data', exist_ok=True)
        
        if format == 'csv':
            data.to_csv(f'data/{self.symbol}_processed.csv')
        elif format == 'json':
            data.to_json(f'data/{self.symbol}_processed.json', orient='records')
        
        print(f"Data for {self.symbol} saved successfully in {format} format")

def main():
    # List of popular equities
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'META', 'NVDA', 'JPM', 'NFLX', 'DIS']
    for symbol in symbols:
        print(f"Processing {symbol}...")
        collector = DataCollector(symbol=symbol)
        raw_data = collector.fetch_data()
        processed_data = collector.preprocess_data(raw_data)
        if processed_data is not None:
            collector.save_data(processed_data, format='csv')
            collector.save_data(processed_data, format='json')
        else:
            print(f"No data for {symbol}.")

if __name__ == "__main__":
    main() 