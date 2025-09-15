import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler

class FeatureEngineer:
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def add_technical_indicators(self, df):
        """
        Add technical indicators to the dataframe with adaptive periods
        
        Parameters:
        df (pandas.DataFrame): Price data with OHLCV columns
        
        Returns:
        pandas.DataFrame: DataFrame with technical indicators
        """
        if df is None or df.empty:
            print("DataFrame is None or empty")
            return None
            
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        print(f"Starting with {len(df_copy)} data points")
        
        # Check and convert column names if they're tuples
        if isinstance(df_copy.columns, pd.MultiIndex):
            df_copy.columns = [col[0] if isinstance(col, tuple) else col for col in df_copy.columns]
        
        # Make sure all column names are lowercase
        df_copy.columns = [col.lower() if isinstance(col, str) else col for col in df_copy.columns]
        
        # Debug: print available columns
        print(f"Available columns: {df_copy.columns.tolist()}")
        
        # Make sure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df_copy.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df_copy.columns]
            print(f"Missing required columns: {missing}")
            return None
        
        # Set Date as index if it's a column
        if 'date' in df_copy.columns:
            try:
                df_copy['date'] = pd.to_datetime(df_copy['date'])
                df_copy = df_copy.set_index('date')
            except Exception as e:
                print(f"Error setting date as index: {e}")
                # Continue without setting date as index
        
        # ADAPTIVE PERIODS based on data size to prevent too many NaN values
        data_length = len(df_copy)
        
        # Calculate adaptive periods (use max 1/4 of data length for longest period)
        short_period = min(5, max(3, data_length // 20))      # ~5% of data
        medium_period = min(20, max(5, data_length // 10))    # ~10% of data  
        long_period = min(50, max(10, data_length // 4))      # ~25% of data
        very_long_period = min(100, max(20, data_length // 3)) # ~33% of data
        
        print(f"Using adaptive periods - Short: {short_period}, Medium: {medium_period}, Long: {long_period}, Very Long: {very_long_period}")
        
        # Add Simple Moving Averages with adaptive periods
        try:
            df_copy['sma_short'] = ta.trend.sma_indicator(df_copy['close'], window=short_period)
            df_copy['sma_medium'] = ta.trend.sma_indicator(df_copy['close'], window=medium_period)
            
            if data_length >= long_period:
                df_copy['sma_long'] = ta.trend.sma_indicator(df_copy['close'], window=long_period)
            
            if data_length >= very_long_period:
                df_copy['sma_very_long'] = ta.trend.sma_indicator(df_copy['close'], window=very_long_period)
        except Exception as e:
            print(f"Warning: SMA indicators failed: {e}")
        
        # Add Exponential Moving Averages with adaptive periods
        try:
            df_copy['ema_short'] = ta.trend.ema_indicator(df_copy['close'], window=short_period)
            df_copy['ema_medium'] = ta.trend.ema_indicator(df_copy['close'], window=medium_period)
            
            if data_length >= long_period:  
                df_copy['ema_long'] = ta.trend.ema_indicator(df_copy['close'], window=long_period)
        except Exception as e:
            print(f"Warning: EMA indicators failed: {e}")
        
        # Add MACD with adaptive periods
        try:
            macd_fast = min(12, max(5, data_length // 8))
            macd_slow = min(26, max(10, data_length // 4))
            macd_signal = min(9, max(3, data_length // 12))
            
            if data_length >= macd_slow:
                df_copy['macd'] = ta.trend.macd(df_copy['close'], window_slow=macd_slow, window_fast=macd_fast)
                df_copy['macd_signal'] = ta.trend.macd_signal(df_copy['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
                df_copy['macd_diff'] = ta.trend.macd_diff(df_copy['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_signal)
        except Exception as e:
            print(f"Warning: MACD indicators failed: {e}")
        
        # Add RSI with adaptive period
        try:
            rsi_period = min(14, max(5, data_length // 7))
            if data_length >= rsi_period:
                df_copy['rsi'] = ta.momentum.rsi(df_copy['close'], window=rsi_period)
        except Exception as e:
            print(f"Warning: RSI indicator failed: {e}")
        
        # Add Bollinger Bands with adaptive period
        try:
            bb_period = min(20, max(5, data_length // 5))
            if data_length >= bb_period:
                bollinger = ta.volatility.BollingerBands(df_copy['close'], window=bb_period)
                df_copy['bollinger_mavg'] = bollinger.bollinger_mavg()
                df_copy['bollinger_hband'] = bollinger.bollinger_hband()
                df_copy['bollinger_lband'] = bollinger.bollinger_lband()
                
                # Add Bollinger Band position (where price sits within bands)
                df_copy['bb_position'] = (df_copy['close'] - df_copy['bollinger_lband']) / (df_copy['bollinger_hband'] - df_copy['bollinger_lband'])
        except Exception as e:
            print(f"Warning: Bollinger Bands failed: {e}")
        
        # Add Stochastic Oscillator with adaptive period
        try:
            stoch_period = min(14, max(5, data_length // 7))
            if data_length >= stoch_period:
                df_copy['stoch'] = ta.momentum.stoch(df_copy['high'], df_copy['low'], df_copy['close'], window=stoch_period)
                df_copy['stoch_signal'] = ta.momentum.stoch_signal(df_copy['high'], df_copy['low'], df_copy['close'], window=stoch_period)
        except Exception as e:
            print(f"Warning: Stochastic indicators failed: {e}")
        
        # Add Average True Range with adaptive period
        try:
            atr_period = min(14, max(3, data_length // 7))
            if data_length >= atr_period:
                df_copy['atr'] = ta.volatility.average_true_range(df_copy['high'], df_copy['low'], df_copy['close'], window=atr_period)
        except Exception as e:
            print(f"Warning: ATR indicator failed: {e}")
        
        # Add OBV (On Balance Volume) - no period needed
        try:
            df_copy['obv'] = ta.volume.on_balance_volume(df_copy['close'], df_copy['volume'])
        except Exception as e:
            print(f"Warning: OBV indicator failed: {e}")
        
        # Add price changes (returns) with multiple periods
        try:
            df_copy['price_change_1d'] = df_copy['close'].pct_change(periods=1)
            
            if data_length >= 5:
                df_copy['price_change_5d'] = df_copy['close'].pct_change(periods=5)
            
            if data_length >= 10:
                df_copy['price_change_10d'] = df_copy['close'].pct_change(periods=10)
        except Exception as e:
            print(f"Warning: Price change indicators failed: {e}")
        
        # Add volume indicators
        try:
            df_copy['volume_sma'] = df_copy['volume'].rolling(window=short_period, min_periods=1).mean()
            df_copy['volume_ratio'] = df_copy['volume'] / df_copy['volume_sma']
            
            # Add additional volume indicators that exist in ta library
            if data_length >= short_period:
                try:
                    df_copy['volume_adi'] = ta.volume.acc_dist_index(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'])
                    df_copy['volume_cmf'] = ta.volume.chaikin_money_flow(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=short_period)
                    df_copy['volume_fi'] = ta.volume.force_index(df_copy['close'], df_copy['volume'], window=short_period)
                    df_copy['volume_mfi'] = ta.volume.money_flow_index(df_copy['high'], df_copy['low'], df_copy['close'], df_copy['volume'], window=short_period)
                except Exception as e:
                    print(f"Warning: Some volume indicators failed to calculate: {e}")
                    # Continue without these indicators
        except Exception as e:
            print(f"Warning: Volume indicators failed: {e}")
        
        # Add price position indicators
        try:
            df_copy['high_low_ratio'] = df_copy['high'] / df_copy['low']
            df_copy['close_open_ratio'] = df_copy['close'] / df_copy['open']
        except Exception as e:
            print(f"Warning: Price ratio indicators failed: {e}")
        
        print(f"Added technical indicators. Shape before NaN handling: {df_copy.shape}")
        print(f"NaN count per column:\n{df_copy.isnull().sum()}")
        
        # SMART NaN handling - use forward fill then backward fill instead of dropping
        df_copy = df_copy.fillna(method='ffill').fillna(method='bfill')
        
        # If there are still NaN values, fill with column means
        for col in df_copy.select_dtypes(include=[np.number]).columns:
            if df_copy[col].isnull().any():
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        
        print(f"Final shape after NaN handling: {df_copy.shape}")
        print(f"Remaining NaN values: {df_copy.isnull().sum().sum()}")
        
        return df_copy
    
    def prepare_features_target(self, df, target_column='close', prediction_horizon=1):
        """
        Prepare features and target for the model with adaptive prediction horizon
        
        Parameters:
        df (pandas.DataFrame): DataFrame with technical indicators
        target_column (str): Column name to predict
        prediction_horizon (int): Number of time periods to predict into the future
        
        Returns:
        tuple: X (features), y_direction (binary target), y_pct_change (percentage change target)
        """
        if df is None or df.empty:
            return None, None, None
            
        df_copy = df.copy()
        data_length = len(df_copy)
        
        # ADAPTIVE prediction horizon based on data size
        # For small datasets, use shorter prediction horizon to preserve more samples
        if data_length < 50:
            prediction_horizon = 1
        elif data_length < 100:
            prediction_horizon = min(prediction_horizon, 2)
        else:
            prediction_horizon = min(prediction_horizon, 5)
        
        print(f"Using prediction horizon: {prediction_horizon} for {data_length} data points")
        
        # Create target variable
        df_copy[f'{target_column}_future'] = df_copy[target_column].shift(-prediction_horizon)
        
        # Create binary target for price direction (1 if price goes up, 0 if down)
        df_copy['target_direction'] = np.where(df_copy[f'{target_column}_future'] > df_copy[target_column], 1, 0)
        
        # Create percentage change target
        df_copy['target_pct_change'] = (df_copy[f'{target_column}_future'] - df_copy[target_column]) / df_copy[target_column]
        
        # Define features (exclude target columns and non-predictive columns)
        exclude_columns = ['target_direction', 'target_pct_change', f'{target_column}_future', 'symbol']
        features = df_copy.drop([col for col in exclude_columns if col in df_copy.columns], axis=1)
        
        # Remove any remaining non-numeric columns except date index
        features = features.select_dtypes(include=[np.number])
        
        # Define targets
        y_direction = df_copy['target_direction']
        y_pct_change = df_copy['target_pct_change']
        
        print(f"Features shape before NaN removal: {features.shape}")
        print(f"Target shapes before NaN removal: direction={y_direction.shape}, pct_change={y_pct_change.shape}")
        
        # Remove rows where targets are NaN (due to shifting)
        # Only remove the last prediction_horizon rows where targets are NaN
        valid_indices = ~(y_direction.isna() | y_pct_change.isna())
        
        features = features[valid_indices]
        y_direction = y_direction[valid_indices]
        y_pct_change = y_pct_change[valid_indices]
        
        print(f"Final shapes after NaN removal:")
        print(f"Features: {features.shape}")
        print(f"Direction target: {y_direction.shape}")
        print(f"Percentage change target: {y_pct_change.shape}")
        
        # Check target distribution
        if len(y_direction) > 0:
            direction_dist = y_direction.value_counts()
            print(f"Direction target distribution: {direction_dist.to_dict()}")
        
        if len(features) == 0:
            print("ERROR: No samples remaining after preprocessing!")
            return None, None, None
        
        return features, y_direction, y_pct_change
    
    def normalize_features(self, features):
        """
        Normalize features using min-max scaling
        
        Parameters:
        features (pandas.DataFrame): Features dataframe
        
        Returns:
        pandas.DataFrame: Normalized features
        """
        if features is None or features.empty:
            print("Features is None or empty for normalization")
            return None
        
        print(f"Normalizing features with shape: {features.shape}")
        
        # Get only numeric columns
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            print("No numeric features found for normalization")
            return None
        
        # Handle any infinite values
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Fit and transform the scaler
        try:
            normalized_features = self.scaler.fit_transform(numeric_features)
            
            # Convert back to DataFrame with original column names
            normalized_df = pd.DataFrame(
                normalized_features, 
                columns=numeric_features.columns, 
                index=numeric_features.index
            )
            
            print(f"Successfully normalized features: {normalized_df.shape}")
            return normalized_df
            
        except Exception as e:
            print(f"Error during normalization: {str(e)}")
            return None

# Test function
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    
    # Fetch some sample data
    fetcher = DataFetcher()
    df = fetcher.get_stock_data("AAPL", period="6mo")
    
    if df is not None:
        engineer = FeatureEngineer()
        
        # Add technical indicators
        df_with_indicators = engineer.add_technical_indicators(df)
        print("\nData with Technical Indicators:")
        print(f"Shape: {df_with_indicators.shape if df_with_indicators is not None else 'None'}")
        
        # Prepare features and target
        X, y_direction, y_pct_change = engineer.prepare_features_target(df_with_indicators)
        print("\nFeatures Shape:", X.shape if X is not None else "None")
        print("Target Direction Shape:", y_direction.shape if y_direction is not None else "None")
        print("Target % Change Shape:", y_pct_change.shape if y_pct_change is not None else "None")
        
        # Normalize features
        if X is not None:
            X_normalized = engineer.normalize_features(X)
            print("\nNormalized Features Shape:", X_normalized.shape if X_normalized is not None else "None")
            if X_normalized is not None:
                print("Sample normalized features:")
                print(X_normalized.head())