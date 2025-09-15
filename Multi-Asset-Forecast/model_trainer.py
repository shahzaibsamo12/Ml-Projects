import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ModelTrainer:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Initialize models
        self.direction_model = None
        self.price_model = None
        self.lstm_model = None
        
    def _validate_data_size(self, X, y, min_samples=10):
        """
        Validate if we have sufficient data for training
        
        Parameters:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        min_samples (int): Minimum required samples
        
        Returns:
        bool: True if data is sufficient, False otherwise
        """
        n_samples = len(X)
        
        if n_samples < 2:
            print(f"Error: Only {n_samples} sample(s) available. Need at least 2 samples for training.")
            print("Suggestions:")
            print("1. Use a shorter interval (1d instead of 1wk)")
            print("2. Use a longer time period (5y instead of 2y)")
            print("3. Check if the symbol has sufficient historical data")
            return False
        
        if n_samples < min_samples:
            print(f"Warning: Only {n_samples} samples available. Recommended minimum is {min_samples}.")
            print("Model performance may be limited with small datasets.")
        
        return True
    
    def _get_optimal_test_size(self, n_samples):
        """
        Get optimal test size based on number of samples
        
        Parameters:
        n_samples (int): Number of samples
        
        Returns:
        float: Optimal test size
        """
        if n_samples < 5:
            return 0.0  # No test set, use all data for training
        elif n_samples < 10:
            return 0.1  # 10% test set
        elif n_samples < 20:
            return 0.15  # 15% test set
        else:
            return 0.2  # Standard 20% test set
        
    def train_direction_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train a model to predict price direction (up or down)
        
        Parameters:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target (binary: 1 for up, 0 for down)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
        Returns:
        tuple: (model, accuracy) or (None, None) if insufficient data
        """
        # Validate data size
        if not self._validate_data_size(X, y):
            return None, None
        
        n_samples = len(X)
        
        # Adjust test size based on available data
        optimal_test_size = self._get_optimal_test_size(n_samples)
        if optimal_test_size < test_size:
            print(f"Adjusting test_size from {test_size} to {optimal_test_size} due to limited data")
            test_size = optimal_test_size
        
        # Handle very small datasets
        if test_size == 0.0:
            print("Using all data for training (no test set due to limited data)")
            X_train = X
            y_train = y
            X_test = X  # Use training data for evaluation (not ideal but functional)
            y_test = y
        else:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
            )
        
        print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Initialize models list
        models_to_train = []
        
        # Only train models that can handle the data size
        if len(X_train) >= 2:
            models_to_train.extend([
                ('Random Forest', RandomForestClassifier(n_estimators=min(100, len(X_train) * 5), random_state=random_state)),
                ('Gradient Boosting', GradientBoostingClassifier(n_estimators=min(100, len(X_train) * 5), random_state=random_state))
            ])
        
        if len(X_train) >= 5:  # Neural networks need more data
            models_to_train.append(
                ('Neural Network', MLPClassifier(hidden_layer_sizes=(min(50, len(X_train)), min(25, len(X_train)//2)), 
                                               max_iter=1000, random_state=random_state))
            )
        
        if not models_to_train:
            print("Error: Insufficient data to train any model")
            return None, None
        
        # Train and evaluate models
        best_model = None
        best_score = 0
        
        for name, model in models_to_train:
            try:
                print(f"Training {name} model for direction prediction...")
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Only calculate other metrics if we have enough diverse predictions
                if len(np.unique(y_test)) > 1 and len(np.unique(y_pred)) > 1:
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                else:
                    precision = recall = f1 = 0.0
                
                print(f"\n{name} Performance:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = model
                    
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
        
        if best_model is None:
            print("Error: No model could be trained successfully")
            return None, None
        
        self.direction_model = best_model
        
        # Save the best model
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(self.models_dir, f'direction_model_{timestamp}.joblib')
            joblib.dump(best_model, model_path)
            print(f"\nBest direction model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {str(e)}")
        
        return best_model, best_score
    
    def train_price_model(self, X, y, test_size=0.2, random_state=42):
        """
        Train a model to predict price change percentage
        
        Parameters:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target (percentage change)
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        
        Returns:
        tuple: (model, r2_score, mean_absolute_error, mean_squared_error) or (None, None, None, None)
        """
        # Validate data size
        if not self._validate_data_size(X, y):
            return None, None, None, None
        
        n_samples = len(X)
        
        # Adjust test size based on available data
        optimal_test_size = self._get_optimal_test_size(n_samples)
        if optimal_test_size < test_size:
            print(f"Adjusting test_size from {test_size} to {optimal_test_size} due to limited data")
            test_size = optimal_test_size
        
        # Handle very small datasets
        if test_size == 0.0:
            print("Using all data for training (no test set due to limited data)")
            X_train = X
            y_train = y
            X_test = X
            y_test = y
        else:
            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        print(f"Training price model with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        try:
            # Train Linear Regression model
            print("Training Linear Regression model for price prediction...")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            # Save the model
            self.price_model = lr
            
            # Calculate metrics
            y_pred = lr.predict(X_test)
            r2 = lr.score(X_test, y_test)
            mae = np.mean(np.abs(y_pred - y_test))
            mse = np.mean((y_pred - y_test)**2)
            
            print(f"\nLinear Regression Performance:")
            print(f"R² Score: {r2:.4f}")
            print(f"Mean Absolute Error: {mae:.4f}")
            print(f"Mean Squared Error: {mse:.4f}")
            
            # Save the model
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(self.models_dir, f'price_model_{timestamp}.joblib')
                joblib.dump(lr, model_path)
                print(f"\nPrice prediction model saved to {model_path}")
            except Exception as e:
                print(f"Warning: Could not save model: {str(e)}")
            
            return lr, r2, mae, mse
            
        except Exception as e:
            print(f"Error training price model: {str(e)}")
            return None, None, None, None

    def train_lstm_model(self, X, y, test_size=0.2, random_state=42, epochs=50, batch_size=32):
        """
        Train an LSTM model for time series prediction
        
        Parameters:
        X (pandas.DataFrame): Features
        y (pandas.Series): Target
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        
        Returns:
        tuple: (model, history) or (None, None) if insufficient data
        """
        # LSTM needs more data than traditional models
        if not self._validate_data_size(X, y, min_samples=20):
            print("LSTM models require at least 20 samples for meaningful training")
            return None, None
        
        n_samples = len(X)
        
        # Adjust parameters for small datasets
        if n_samples < 50:
            epochs = min(epochs, 20)
            batch_size = min(batch_size, max(1, n_samples // 4))
            print(f"Adjusting LSTM parameters for small dataset: epochs={epochs}, batch_size={batch_size}")
        
        # Convert DataFrame to numpy arrays
        X_values = X.values
        y_values = y.values
        
        # Reshape data for LSTM [samples, time steps, features]
        X_values = X_values.reshape((X_values.shape[0], 1, X_values.shape[1]))
        
        # Adjust test size
        optimal_test_size = self._get_optimal_test_size(n_samples)
        if optimal_test_size < test_size:
            test_size = optimal_test_size
        
        # Split data into train and test sets
        if test_size == 0.0:
            X_train = X_test = X_values
            y_train = y_test = y_values
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_values, y_values, test_size=test_size, random_state=random_state)
        
        try:
            # Create LSTM model with adjusted architecture for small datasets
            model = Sequential()
            lstm_units = min(50, max(10, n_samples // 4))
            dense_units = min(25, max(5, n_samples // 8))
            
            model.add(LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(lstm_units, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(dense_units))
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
            
            # Define early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=min(10, epochs//5), restore_best_weights=True)
            
            # Train the model
            print(f"Training LSTM model with {len(X_train)} samples...")
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stop],
                verbose=1
            )
            
            # Save the model
            self.lstm_model = model
            
            # Save the keras model
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(self.models_dir, f'lstm_model_{timestamp}')
                model.save(model_path)
                print(f"\nLSTM model saved to {model_path}")
            except Exception as e:
                print(f"Warning: Could not save LSTM model: {str(e)}")
            
            return model, history
            
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            return None, None
    
    def load_models(self, direction_model_path, price_model_path=None, lstm_model_path=None):
        """
        Load pre-trained models
        
        Parameters:
        direction_model_path (str): Path to direction prediction model
        price_model_path (str): Path to price prediction model
        lstm_model_path (str): Path to LSTM model
        """
        # Load direction model
        if os.path.exists(direction_model_path):
            self.direction_model = joblib.load(direction_model_path)
            print(f"Direction model loaded from {direction_model_path}")
        
        # Load price model
        if price_model_path and os.path.exists(price_model_path):
            self.price_model = joblib.load(price_model_path)
            print(f"Price model loaded from {price_model_path}")
        
        # Load LSTM model
        if lstm_model_path and os.path.exists(lstm_model_path):
            self.lstm_model = tf.keras.models.load_model(lstm_model_path)
            print(f"LSTM model loaded from {lstm_model_path}")
    
    def predict_direction(self, X):
        """
        Predict price direction
        
        Parameters:
        X (pandas.DataFrame): Features
        
        Returns:
        numpy.ndarray: Predictions (1 for up, 0 for down)
        """
        if self.direction_model is None:
            print("Direction model not trained or loaded.")
            return None
        
        return self.direction_model.predict(X)
    
    def predict_price_change(self, X):
        """
        Predict price change percentage
        
        Parameters:
        X (pandas.DataFrame): Features
        
        Returns:
        numpy.ndarray: Price change predictions
        """
        if self.price_model is None:
            print("Price model not trained or loaded.")
            return None
        
        return self.price_model.predict(X)
    
    def predict_lstm(self, X):
        """
        Make predictions using LSTM model
        
        Parameters:
        X (pandas.DataFrame): Features
        
        Returns:
        numpy.ndarray: Predictions
        """
        if self.lstm_model is None:
            print("LSTM model not trained or loaded.")
            return None
        
        # Reshape data for LSTM
        X_values = X.values
        X_values = X_values.reshape((X_values.shape[0], 1, X_values.shape[1]))
        
        return self.lstm_model.predict(X_values)

# Test function
if __name__ == "__main__":
    from data_fetcher import DataFetcher
    from feature_engineering import FeatureEngineer
    
    # Fetch sample data
    fetcher = DataFetcher()
    df = fetcher.get_stock_data("AAPL", period="2y")
    
    if df is not None:
        # Prepare features
        engineer = FeatureEngineer()
        df_with_indicators = engineer.add_technical_indicators(df)
        X, y_direction, y_pct_change = engineer.prepare_features_target(df_with_indicators)
        X_normalized = engineer.normalize_features(X)
        
        if X_normalized is not None and y_direction is not None and y_pct_change is not None:
            # Train models
            trainer = ModelTrainer()
            direction_result = trainer.train_direction_model(X_normalized, y_direction)
            price_result = trainer.train_price_model(X_normalized, y_pct_change)
            
            # Check if training was successful
            if direction_result[0] is not None:
                # Make predictions
                direction_pred = trainer.predict_direction(X_normalized.iloc[-5:])
                print("\nSample Direction Predictions:")
                print(direction_pred)
            
            if price_result[0] is not None:
                price_pred = trainer.predict_price_change(X_normalized.iloc[-5:])
                print("\nSample Price Change Predictions:")
                print(price_pred)