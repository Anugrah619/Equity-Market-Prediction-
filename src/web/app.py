from flask import Flask, render_template, jsonify, request
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime, timedelta
import yfinance as yf
import os

# Initialize Flask app
app = Flask(__name__)

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# List of available equities
EQUITIES = [
    {'label': 'Apple (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft (MSFT)', 'value': 'MSFT'},
    {'label': 'Alphabet (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Tesla (TSLA)', 'value': 'TSLA'},
    {'label': 'Amazon (AMZN)', 'value': 'AMZN'},
    {'label': 'Meta (META)', 'value': 'META'},
    {'label': 'NVIDIA (NVDA)', 'value': 'NVDA'},
    {'label': 'JPMorgan Chase (JPM)', 'value': 'JPM'},
    {'label': 'Netflix (NFLX)', 'value': 'NFLX'},
    {'label': 'Disney (DIS)', 'value': 'DIS'},
]

def get_latest_data(symbol='^GSPC', days=90):
    """Fetch latest market data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def prepare_features(data):
    """Prepare features for prediction"""
    df = data.copy()
    # Ensure columns are unique and correct
    df = df.loc[:, ~df.columns.duplicated()]
    # Ensure 'Close' and 'Volume' are Series
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    volume = df['Volume'] if isinstance(df['Volume'], pd.Series) else df['Volume'].iloc[:, 0]
    # Calculate technical indicators
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    exp1 = close.ewm(span=12, adjust=False).mean()
    exp2 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df['BB_Middle'] = close.rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * close.rolling(window=20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * close.rolling(window=20).std()
    # Volume indicators
    df['Volume_SMA'] = volume.rolling(window=20).mean()
    # Fill any remaining NaNs (forward then backward fill)
    df = df.ffill().bfill()
    return df

def load_equity_model_scaler_metrics(symbol):
    """Load model, scaler, and metrics for a specific equity"""
    try:
        model_path = f'models/{symbol}_model.joblib'
        scaler_path = f'models/{symbol}_scaler.joblib'
        metrics_path = f'models/{symbol}_metrics.json'
        
        if not all(os.path.exists(path) for path in [model_path, scaler_path, metrics_path]):
            print(f"Missing files for {symbol}")
            return None, None, None
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        return model, scaler, metrics
    except Exception as e:
        print(f"Error loading model/scaler/metrics for {symbol}: {e}")
        return None, None, None

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['GET'])
def predict():
    symbol = request.args.get('symbol', 'AAPL')
    model, scaler, _ = load_equity_model_scaler_metrics(symbol)
    
    if model is None or scaler is None:
        return jsonify({'error': f'Model not trained for {symbol}'})
        
    try:
        data = get_latest_data(symbol)
        processed_data = prepare_features(data)
        
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 
                   'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 
                   'BB_Upper', 'BB_Lower', 'Volume_SMA']
        
        X = processed_data[features].iloc[-1:]
        if X.isnull().any().any():
            return jsonify({'error': f'Prediction not possible for {symbol}: latest data contains missing values after feature engineering.'})
        X_scaled = scaler.transform(X.values)
        prediction = model.predict(X_scaled)[0]
        
        return jsonify({
            'prediction': float(prediction),
            'current_price': float(data['Close'].iloc[-1]),
            'predicted_return': float(prediction),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': f'Prediction failed for {symbol}: {str(e)}'})

# Dash layout
dash_app.layout = html.Div([
    html.H1('Equity Market Prediction Dashboard'),
    
    # Equity Dropdown
    html.Div([
        html.Label('Select Equity:'),
        dcc.Dropdown(
            id='equity-dropdown',
            options=EQUITIES,
            value='AAPL',
            clearable=False,
            style={'width': '300px'}
        )
    ], style={'marginBottom': 30}),
    
    # Model Performance Metrics
    html.Div([
        html.H2('Model Performance Metrics'),
        html.Div(id='metrics-display')
    ]),
    
    # Price Chart
    html.Div([
        html.H2('Price Chart'),
        dcc.Graph(id='price-chart')
    ]),
    
    # Prediction Chart
    html.Div([
        html.H2('Prediction Chart'),
        dcc.Graph(id='prediction-chart')
    ]),
    
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # 5 minutes
        n_intervals=0
    )
])

# Dash callbacks
@dash_app.callback(
    [Output('price-chart', 'figure'),
     Output('prediction-chart', 'figure'),
     Output('metrics-display', 'children')],
    [Input('equity-dropdown', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_graphs(selected_equity, n):
    # Load processed data for the selected equity
    try:
        df = pd.read_csv(f'data/{selected_equity}_processed.csv', parse_dates=['Date'])
        print(f"Loaded {selected_equity} data shape: {df.shape}")
    except Exception as e:
        print(f"Error loading data for {selected_equity}: {e}")
        return go.Figure(), go.Figure(), html.P(f"No data for {selected_equity}")
        
    # Load model, scaler, and metrics
    model, scaler, metrics = load_equity_model_scaler_metrics(selected_equity)
    if model is None or scaler is None:
        print(f"Model or scaler not found for {selected_equity}")
        return go.Figure(), go.Figure(), html.P(f"Model not trained for {selected_equity}")
    
    # Price chart
    price_fig = go.Figure()
    price_fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
    price_fig.update_layout(
        title=f'Price Chart - {selected_equity}',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    # Prediction chart
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 
               'SMA_50', 'RSI', 'MACD', 'Signal_Line', 'BB_Middle', 
               'BB_Upper', 'BB_Lower', 'Volume_SMA']
    # Drop rows with NaNs in features
    df_pred = df.dropna(subset=features)
    print(f"{selected_equity} df_pred shape after dropna: {df_pred.shape}")
    if 'Target' not in df_pred.columns:
        print(f"Target column missing in {selected_equity} data!")
        return price_fig, go.Figure(), html.P(f"No Target column for {selected_equity}")
    if df_pred.empty:
        print(f"No valid data for predictions for {selected_equity}")
        return price_fig, go.Figure(), html.P(f"No valid data for predictions for {selected_equity}")
    if df_pred['Target'].isnull().all():
        print(f"All Target values are NaN for {selected_equity}")
        return price_fig, go.Figure(), html.P(f"All Target values are NaN for {selected_equity}")
    X = df_pred[features].values
    y_true = df_pred['Target'].values
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    
    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(
        x=df_pred['Date'],
        y=y_pred,
        name='Predicted Return',
        line=dict(color='blue')
    ))
    pred_fig.add_trace(go.Scatter(
        x=df_pred['Date'],
        y=y_true,
        name='Actual Return',
        line=dict(color='red', dash='dot')
    ))
    pred_fig.update_layout(
        title=f'Predicted vs Actual Returns - {selected_equity}',
        xaxis_title='Date',
        yaxis_title='Return',
        template='plotly_white'
    )
    
    # Model performance metrics (regression)
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{selected_equity} metrics: MAE={mae}, RMSE={rmse}, R2={r2}")
    
    # Up/down classification metrics
    y_true_class = (y_true > 0).astype(int)
    y_pred_class = (y_pred > 0).astype(int)
    accuracy = accuracy_score(y_true_class, y_pred_class)
    precision = precision_score(y_true_class, y_pred_class, zero_division=0)
    recall = recall_score(y_true_class, y_pred_class, zero_division=0)
    
    metrics_display = html.Div([
        html.P(f"Model: {metrics.get('model_name', 'Unknown')}"),
        html.P(f"MAE: {mae:.4f}"),
        html.P(f"RMSE: {rmse:.4f}"),
        html.P(f"R² Score: {r2:.4f}"),
        html.P(f"Accuracy (Up/Down): {accuracy:.4f}"),
        html.P(f"Precision (Up/Down): {precision:.4f}"),
        html.P(f"Recall (Up/Down): {recall:.4f}"),
        html.P(f"(You can quote these metrics for {selected_equity} in your resume)")
    ])
    
    return price_fig, pred_fig, metrics_display

if __name__ == '__main__':
    app.run(debug=True) 