import pandas as pd
import numpy as np
from datetime import datetime

class SignalGenerator:
    def __init__(self, direction_model=None, price_model=None, lstm_model=None):
        self.direction_model = direction_model
        self.price_model = price_model
        self.lstm_model = lstm_model
        
    def set_models(self, direction_model=None, price_model=None, lstm_model=None):
        """
        Set models for signal generation
        
        Parameters:
        direction_model: Model to predict price direction
        price_model: Model to predict price change
        lstm_model: LSTM model for time series prediction
        """
        if direction_model is not None:
            self.direction_model = direction_model
        if price_model is not None:
            self.price_model = price_model
        if lstm_model is not None:
            self.lstm_model = lstm_model
    
    def generate_signals(self, X, current_price):
        """
        Generate trading signals based on model predictions
        
        Parameters:
        X (pandas.DataFrame): Features
        current_price (float): Current price
        
        Returns:
        dict: Trading signals and predictions
        """
        signals = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'current_price': current_price,
            'signal': 'HOLD',  # Default signal
            'confidence': 0.0,
            'target_price': current_price,
            'stop_loss': current_price,
            'take_profit': current_price,
            'prediction': {}
        }
        
        # Get direction prediction
        if self.direction_model is not None:
            direction_pred = self.direction_model.predict(X)
            direction_prob = self.direction_model.predict_proba(X) if hasattr(self.direction_model, 'predict_proba') else None
            
            signals['prediction']['direction'] = int(direction_pred[-1])
            signals['prediction']['direction_probability'] = float(direction_prob[-1, 1]) if direction_prob is not None else None
            
            # Set confidence based on probability
            if direction_prob is not None:
                confidence = direction_prob[-1, 1] if direction_pred[-1] == 1 else direction_prob[-1, 0]
                signals['confidence'] = float(confidence)
        
        # Get price change prediction
        if self.price_model is not None:
            price_change_pred = self.price_model.predict(X)
            signals['prediction']['price_change'] = float(price_change_pred[-1])
            
            # Calculate target price
            target_price = current_price * (1 + price_change_pred[-1])
            signals['target_price'] = float(target_price)
        
        # Determine signal
        if signals['prediction'].get('direction') == 1 and signals['confidence'] >= 0.65:
            signals['signal'] = 'BUY'
            signals['stop_loss'] = float(current_price * 0.97)  # 3% stop loss
            signals['take_profit'] = float(current_price * (1 + signals['prediction'].get('price_change', 0.05)))
        elif signals['prediction'].get('direction') == 0 and signals['confidence'] >= 0.65:
            signals['signal'] = 'SELL'
            signals['stop_loss'] = float(current_price * 1.03)  # 3% stop loss
            signals['take_profit'] = float(current_price * (1 - signals['prediction'].get('price_change', 0.05)))
        
        return signals
    
    def generate_signals_for_dataframe(self, df, features, feature_engineer, price_column='close'):
        """
        Generate signals for a dataframe
    
        Parameters:
        df (pandas.DataFrame): Price data with features
        features (pandas.DataFrame): Normalized features
        feature_engineer (FeatureEngineer): Feature engineering object
        price_column (str): Column name for price data
    
        Returns:
        pandas.DataFrame: DataFrame with signals
        """
        if df is None or df.empty or features is None or features.empty:
            print("Either df or features is None or empty")
            return None
    
        # Make a copy of the dataframe
        result_df = df.copy()
    
        # Ensure price column exists
        if price_column not in result_df.columns:
            print(f"Price column '{price_column}' not found in dataframe. Available columns: {result_df.columns.tolist()}")
            return None
    
        # Print some debugging info
        print(f"Result DataFrame shape: {result_df.shape}")
        print(f"Features shape: {features.shape}")
    
        # Initialize signal columns
        result_df['signal'] = 'HOLD'
        result_df['confidence'] = 0.0
        result_df['target_price'] = result_df[price_column]
        result_df['stop_loss'] = result_df[price_column]
        result_df['take_profit'] = result_df[price_column]
    
        # Make sure the indexes match or can be aligned
        try:
            # Try to align indexes
            common_index = result_df.index.intersection(features.index)
        
            if len(common_index) == 0:
                print("No common index values found between DataFrames. Trying to reset indexes.")
                # If no common indices, try with reset_index
                result_df = result_df.reset_index()
                features_reset = features.reset_index()
            
                # Use row positions instead
                result_df = result_df.iloc[:min(len(result_df), len(features))]
                features_aligned = features_reset.iloc[:len(result_df)]
            else:
                # Use common indexes
                result_df = result_df.loc[common_index]
                features_aligned = features.loc[common_index]
            
            print(f"Aligned shapes - result_df: {result_df.shape}, features: {features_aligned.shape}")
        
            # Generate predictions for direction
            if self.direction_model is not None:
                direction_pred = self.direction_model.predict(features_aligned)
                result_df['pred_direction'] = direction_pred
            
                if hasattr(self.direction_model, 'predict_proba'):
                    direction_prob = self.direction_model.predict_proba(features_aligned)
                    result_df['direction_prob_up'] = direction_prob[:, 1]
                    result_df['direction_prob_down'] = direction_prob[:, 0]
                
                    # Set confidence
                    result_df['confidence'] = np.where(
                        result_df['pred_direction'] == 1,
                        result_df['direction_prob_up'],
                        result_df['direction_prob_down']
                    )
        
            # Generate predictions for price change
            if self.price_model is not None:
                price_change_pred = self.price_model.predict(features_aligned)
                result_df['pred_price_change'] = price_change_pred
            
                # Calculate target price
                result_df['target_price'] = result_df[price_column] * (1 + result_df['pred_price_change'])
        
            # Determine signals
            high_confidence_mask = result_df['confidence'] >= 0.65
        
            # BUY signals
            buy_mask = (result_df['pred_direction'] == 1) & high_confidence_mask
            result_df.loc[buy_mask, 'signal'] = 'BUY'
            result_df.loc[buy_mask, 'stop_loss'] = result_df.loc[buy_mask, price_column] * 0.97
            result_df.loc[buy_mask, 'take_profit'] = result_df.loc[buy_mask, price_column] * (1 + result_df.loc[buy_mask, 'pred_price_change'])
        
            # SELL signals
            sell_mask = (result_df['pred_direction'] == 0) & high_confidence_mask
            result_df.loc[sell_mask, 'signal'] = 'SELL'
            result_df.loc[sell_mask, 'stop_loss'] = result_df.loc[sell_mask, price_column] * 1.03
            result_df.loc[sell_mask, 'take_profit'] = result_df.loc[sell_mask, price_column] * (1 - result_df.loc[sell_mask, 'pred_price_change'])
        
            return result_df
        except Exception as e:
            print(f"Error aligning DataFrames: {e}")
            return None