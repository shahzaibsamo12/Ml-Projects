import pandas as pd
import numpy as np
import os
import joblib
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from data_fetcher import DataFetcher
from feature_engineering import FeatureEngineer
from model_trainer import ModelTrainer
from signal_generator import SignalGenerator

class PredictionSystem:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.data_fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.model_trainer = ModelTrainer(models_dir=models_dir)
        self.signal_generator = SignalGenerator()
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Initialize models
        self.direction_model_path = None
        self.price_model_path = None
        self.lstm_model_path = None
        
        # Initialize latest data and predictions
        self.latest_data = {}
        self.latest_predictions = {}
    
    def train_models(self, symbol, asset_type='stock', period='2y', interval='1d'):
        """
        Train prediction models for a specific asset
        
        Parameters:
        symbol (str): Asset symbol (e.g., 'AAPL', 'EURUSD', 'BTC-USD')
        asset_type (str): Type of asset ('stock', 'forex', 'crypto')
        period (str): Period to fetch data
        interval (str): Data interval
        
        Returns:
        bool: True if training succeeded, False otherwise
        """
        print(f"\nTraining models for {symbol} ({asset_type})...")
        
        # Fetch historical data
        df = None
        if asset_type == 'stock':
            df = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
        elif asset_type == 'forex':
            df = self.data_fetcher.get_forex_data(symbol, period=period, interval=interval)
        elif asset_type == 'crypto':
            df = self.data_fetcher.get_crypto_data(symbol, period=period, interval=interval)
        
        if df is None or df.empty:
            print(f"Failed to fetch data for {symbol}")
            return False
        
        print(f"Fetched {len(df)} data points for {symbol}")
        
        # Add technical indicators
        df_with_indicators = self.feature_engineer.add_technical_indicators(df)
        if df_with_indicators is None or df_with_indicators.empty:
            print("Failed to add technical indicators")
            return False
        
        # Prepare features and target
        X, y_direction, y_pct_change = self.feature_engineer.prepare_features_target(df_with_indicators)
        if X is None or y_direction is None or y_pct_change is None:
            print("Failed to prepare features and target")
            return False
        
        print(f"Prepared {len(X)} feature samples")
        
        # Normalize features
        X_normalized = self.feature_engineer.normalize_features(X)
        if X_normalized is None:
            print("Failed to normalize features")
            return False
        
        # Train direction model - FIXED to handle new return format
        direction_result = self.model_trainer.train_direction_model(X_normalized, y_direction)
        if direction_result is None or direction_result[0] is None:
            print("Failed to train direction model")
            return False
        
        direction_model, direction_score = direction_result
        print(f"Direction model trained with accuracy: {direction_score:.4f}")
        
        # Train price model - FIXED to handle new return format
        price_result = self.model_trainer.train_price_model(X_normalized, y_pct_change)
        if price_result is None or price_result[0] is None:
            print("Failed to train price model")
            return False
        
        price_model, r2_score, mae, mse = price_result
        print(f"Price model trained with R² score: {r2_score:.4f}")
        
        # Set model paths
        self.direction_model_path = os.path.join(self.models_dir, f'direction_model_{symbol}.joblib')
        self.price_model_path = os.path.join(self.models_dir, f'price_model_{symbol}.joblib')
        
        # Save models specifically for this symbol
        try:
            joblib.dump(direction_model, self.direction_model_path)
            joblib.dump(price_model, self.price_model_path)
            print(f"Models for {symbol} saved successfully")
        except Exception as e:
            print(f"Warning: Could not save models for {symbol}: {str(e)}")
            # Continue anyway as models are still in memory
        
        # Set models for signal generator
        self.signal_generator.set_models(direction_model, price_model)
        
        return True
    
    def load_models(self, symbol):
        """
        Load pre-trained models for a specific symbol
        
        Parameters:
        symbol (str): Asset symbol
        
        Returns:
        bool: True if loading succeeded, False otherwise
        """
        direction_model_path = os.path.join(self.models_dir, f'direction_model_{symbol}.joblib')
        price_model_path = os.path.join(self.models_dir, f'price_model_{symbol}.joblib')
        
        if not os.path.exists(direction_model_path) or not os.path.exists(price_model_path):
            print(f"Models for {symbol} not found. Please train models first.")
            return False
        
        try:
            self.model_trainer.load_models(direction_model_path, price_model_path)
            
            # Check if models were loaded successfully
            if self.model_trainer.direction_model is None or self.model_trainer.price_model is None:
                print(f"Failed to load models for {symbol}")
                return False
            
            self.signal_generator.set_models(self.model_trainer.direction_model, self.model_trainer.price_model)
            
            self.direction_model_path = direction_model_path
            self.price_model_path = price_model_path
            
            print(f"Models for {symbol} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models for {symbol}: {e}")
            return False
    
    def get_prediction(self, symbol, asset_type='stock', period='1mo', interval='1d'):
        """
        Get predictions for a specific asset
        
        Parameters:
        symbol (str): Asset symbol
        asset_type (str): Type of asset ('stock', 'forex', 'crypto')
        period (str): Period to fetch data
        interval (str): Data interval
        
        Returns:
        dict: Prediction results
        """
        # Fetch latest data
        df = None
        if asset_type == 'stock':
            df = self.data_fetcher.get_stock_data(symbol, period=period, interval=interval)
        elif asset_type == 'forex':
            df = self.data_fetcher.get_forex_data(symbol, period=period, interval=interval)
        elif asset_type == 'crypto':
            df = self.data_fetcher.get_crypto_data(symbol, period=period, interval=interval)
        
        if df is None or df.empty:
            print(f"Failed to fetch data for {symbol}")
            return {'error': f"Failed to fetch data for {symbol}"}
        
        # Check for minimum data requirement
        if len(df) < 2:
            return {'error': f"Insufficient data for {symbol}. Only {len(df)} data points available. Try using a different interval or longer period."}
        
        # Add technical indicators
        df_with_indicators = self.feature_engineer.add_technical_indicators(df)
        if df_with_indicators is None or df_with_indicators.empty:
            print("Failed to add technical indicators")
            return {'error': "Failed to add technical indicators"}
        
        # Prepare features
        X, _, _ = self.feature_engineer.prepare_features_target(df_with_indicators)
        if X is None:
            print("Failed to prepare features")
            return {'error': "Failed to prepare features"}
        
        # Check feature availability
        if len(X) < 1:
            return {'error': "No valid features could be prepared from the data"}
        
        # Normalize features
        X_normalized = self.feature_engineer.normalize_features(X)
        if X_normalized is None:
            print("Failed to normalize features")
            return {'error': "Failed to normalize features"}
        
        # Try to load models if not already loaded
        if self.model_trainer.direction_model is None or self.model_trainer.price_model is None:
            if not self.load_models(symbol):
                # If models don't exist, train them with a longer period for better training data
                if period in ['1mo', '3mo', '6mo']:
                    training_period = '2y'  # Use at least 2 years for training
                elif period == '1y':
                    training_period = '2y'  # Use 2 years for training
                else:
                    training_period = period  # Use selected period if it's already long
                
                print(f"Training new models for {symbol} using {training_period} period (selected: {period})...")
                if not self.train_models(symbol, asset_type, training_period, interval):
                    return {'error': f"Failed to train models for {symbol}. This could be due to insufficient data."}
        
        # Check if models are still None after loading/training
        if self.model_trainer.direction_model is None or self.model_trainer.price_model is None:
            return {'error': f"Models for {symbol} are not available. Training may have failed due to insufficient data."}
        
        # Generate signals for the latest data
        try:
            latest_features = X_normalized.iloc[-1].to_frame().transpose()
            latest_price = df_with_indicators.iloc[-1]['close']
            
            signals = self.signal_generator.generate_signals(latest_features, latest_price)
            
            # Generate signals for entire dataframe for visualization
            signals_df = self.signal_generator.generate_signals_for_dataframe(df_with_indicators, X_normalized, self.feature_engineer)
            
            # Store latest data and predictions
            self.latest_data[symbol] = {
                'df': df,
                'df_with_indicators': df_with_indicators,
                'features': X_normalized,
                'signals_df': signals_df,
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.latest_predictions[symbol] = signals
            
            # Return prediction results
            return {
                'symbol': symbol,
                'asset_type': asset_type,
                'current_price': float(latest_price),
                'signal': signals['signal'],
                'confidence': float(signals['confidence']),
                'target_price': float(signals['target_price']),
                'stop_loss': float(signals['stop_loss']),
                'take_profit': float(signals['take_profit']),
                'prediction': signals['prediction'],
                'last_update': signals['timestamp']
            }
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return {'error': f"Error generating prediction signals: {str(e)}"}
    
    def get_multiple_predictions(self, symbols, asset_types):
        """
        Get predictions for multiple assets
        
        Parameters:
        symbols (list): List of asset symbols
        asset_types (list): List of asset types
        
        Returns:
        dict: Predictions for all assets
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            asset_type = asset_types[i] if i < len(asset_types) else 'stock'
            results[symbol] = self.get_prediction(symbol, asset_type)
        
        return results
    
    def get_historical_signals(self, symbol, asset_type='stock', period='6mo', interval='1d'):
        """
        Get historical signals for backtesting
        
        Parameters:
        symbol (str): Asset symbol
        asset_type (str): Type of asset
        period (str): Period to fetch data
        interval (str): Data interval
        
        Returns:
        pandas.DataFrame: DataFrame with historical signals
        """
        # Check if we already have the data
        if symbol in self.latest_data and 'signals_df' in self.latest_data[symbol]:
            return self.latest_data[symbol]['signals_df']
        
        # Otherwise, get new prediction which will populate latest_data
        prediction = self.get_prediction(symbol, asset_type, period, interval)
        
        # Check if prediction was successful
        if 'error' in prediction:
            print(f"Error getting historical signals: {prediction['error']}")
            return None
        
        if symbol in self.latest_data and 'signals_df' in self.latest_data[symbol]:
            return self.latest_data[symbol]['signals_df']
        
        return None
    
    def backtest_strategy(self, symbol, asset_type='stock', period='1y', interval='1d'):
        """
        Backtest the trading strategy
        
        Parameters:
        symbol (str): Asset symbol
        asset_type (str): Type of asset
        period (str): Period to fetch data
        interval (str): Data interval
        
        Returns:
        dict: Backtest results
        """
        # Get historical signals
        signals_df = self.get_historical_signals(symbol, asset_type, period, interval)
        
        if signals_df is None or signals_df.empty:
            return {'error': f"Failed to get historical signals for {symbol}"}
        
        # Initialize portfolio
        initial_capital = 10000.0
        position = 0
        capital = initial_capital
        portfolio_value = []
        trades = []
        
        # Iterate through the signals
        for i, row in signals_df.iterrows():
            current_portfolio_value = capital + position * row['close']
            portfolio_value.append(current_portfolio_value)
            
            # Execute BUY signal
            if row['signal'] == 'BUY' and position == 0:
                position = capital / row['close']
                capital = 0
                trades.append({
                    'date': row.name if isinstance(row.name, str) else str(row.name),
                    'action': 'BUY',
                    'price': row['close'],
                    'position': position,
                    'capital': capital,
                    'portfolio_value': position * row['close']
                })
            
            # Execute SELL signal
            elif row['signal'] == 'SELL' and position > 0:
                capital = position * row['close']
                position = 0
                trades.append({
                    'date': row.name if isinstance(row.name, str) else str(row.name),
                    'action': 'SELL',
                    'price': row['close'],
                    'position': position,
                    'capital': capital,
                    'portfolio_value': capital
                })
        
        # Calculate final portfolio value
        final_value = capital + (position * signals_df['close'].iloc[-1])
        
        # Calculate metrics
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Add portfolio value to signals dataframe
        signals_df = signals_df.copy()  # Avoid modifying original
        signals_df['portfolio_value'] = portfolio_value
        
        # Return backtest results
        results = {
            'symbol': symbol,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(trades),
            'trades': trades,
            'signals_df': signals_df
        }
        
        return results
    
    def plot_signals(self, symbol, asset_type='stock', period='6mo', interval='1d'):
        """
        Plot price chart with trading signals
        
        Parameters:
        symbol (str): Asset symbol
        asset_type (str): Type of asset
        period (str): Period to fetch data
        interval (str): Data interval
        
        Returns:
        tuple: Figure objects
        """
        # Get historical signals
        signals_df = self.get_historical_signals(symbol, asset_type, period, interval)
        
        if signals_df is None or signals_df.empty:
            print(f"Failed to get historical signals for {symbol}")
            return None
        
        # Create price chart with signals
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot close price
        ax.plot(signals_df.index, signals_df['close'], label='Close Price', color='blue', alpha=0.7)
        
        # Plot buy signals
        buy_signals = signals_df[signals_df['signal'] == 'BUY']
        if not buy_signals.empty:
            ax.scatter(buy_signals.index, buy_signals['close'], color='green', label='Buy Signal', marker='^', s=100)
        
        # Plot sell signals
        sell_signals = signals_df[signals_df['signal'] == 'SELL']
        if not sell_signals.empty:
            ax.scatter(sell_signals.index, sell_signals['close'], color='red', label='Sell Signal', marker='v', s=100)
        
        # Add title and labels
        ax.set_title(f'{symbol} Price with Trading Signals')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig

# Test function
if __name__ == "__main__":
    # Create prediction system
    system = PredictionSystem()
    
    # Test prediction for a stock
    prediction = system.get_prediction("AAPL", asset_type='stock')
    print("\nStock Prediction:")
    print(prediction)
    
    # Test prediction for a forex pair
    prediction = system.get_prediction("EURUSD", asset_type='forex')
    print("\nForex Prediction:")
    print(prediction)
    
    # Test prediction for a cryptocurrency
    prediction = system.get_prediction("BTC-USD", asset_type='crypto')
    print("\nCrypto Prediction:")
    print(prediction)
    
    # Backtest the strategy
    backtest = system.backtest_strategy("AAPL", asset_type='stock')
    print("\nBacktest Results:")
    if 'error' not in backtest:
        print(f"Initial Capital: ${backtest['initial_capital']:.2f}")
        print(f"Final Value: ${backtest['final_value']:.2f}")
        print(f"Total Return: {backtest['total_return']:.2f}%")
        print(f"Total Trades: {backtest['total_trades']}")
    else:
        print(f"Backtest Error: {backtest['error']}")