import streamlit as st

# Set page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Stock Market Prediction System",
    page_icon="📈",
    layout="wide"
)

import traceback
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import os

try:
    st.title("Stock Market Prediction System")
    
    # Try importing custom modules
    try:
        from data_fetcher import DataFetcher
        st.write("✅ DataFetcher imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import DataFetcher: {str(e)}")
        st.code(traceback.format_exc())
    
    try:
        from feature_engineering import FeatureEngineer
        st.write("✅ FeatureEngineer imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import FeatureEngineer: {str(e)}")
        st.code(traceback.format_exc())
    
    try:
        from model_trainer import ModelTrainer
        st.write("✅ ModelTrainer imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import ModelTrainer: {str(e)}")
        st.code(traceback.format_exc())
    
    try:
        from signal_generator import SignalGenerator
        st.write("✅ SignalGenerator imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import SignalGenerator: {str(e)}")
        st.code(traceback.format_exc())
    
    try:
        from prediction_system import PredictionSystem
        st.write("✅ PredictionSystem imported successfully")
    except Exception as e:
        st.error(f"❌ Failed to import PredictionSystem: {str(e)}")
        st.code(traceback.format_exc())
    
    # Define functions for UI with error handling
    def display_prediction(prediction):
        try:
            if 'error' in prediction:
                st.error(prediction['error'])
                return
            
            # Display basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"${prediction['current_price']:.2f}")
            with col2:
                signal_color = "green" if prediction['signal'] == 'BUY' else "red" if prediction['signal'] == 'SELL' else "gray"
                st.markdown(f"<h1 style='text-align: center; color: {signal_color};'>{prediction['signal']}</h1>", unsafe_allow_html=True)
            with col3:
                st.metric("Confidence", f"{prediction['confidence']*100:.2f}%")
            
            # Display price targets
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Price", f"${prediction['target_price']:.2f}")
            with col2:
                st.metric("Stop Loss", f"${prediction['stop_loss']:.2f}")
            with col3:
                st.metric("Take Profit", f"${prediction['take_profit']:.2f}")
            
            # Display additional prediction details
            st.subheader("Prediction Details")
            pred_df = pd.DataFrame({
                "Metric": ["Direction", "Direction Probability", "Price Change %"],
                "Value": [
                    "UP" if prediction['prediction'].get('direction') == 1 else "DOWN",
                    f"{prediction['prediction'].get('direction_probability', 0)*100:.2f}%",
                    f"{prediction['prediction'].get('price_change', 0)*100:.2f}%"
                ]
            })
            st.dataframe(pred_df, hide_index=True)
            
            # Display last update time
            st.caption(f"Last updated: {prediction['last_update']}")
        except Exception as e:
            st.error(f"Error in display_prediction: {e}")
            st.code(traceback.format_exc())

    def plot_price_chart(symbol, asset_type, period, interval):
        try:
            # Get historical signals
            signals_df = system.get_historical_signals(symbol, asset_type, period, interval)
            
            if signals_df is None or signals_df.empty:
                st.warning(f"No historical data available for {symbol}")
                return
            
            # Create plotly chart
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                              vertical_spacing=0.1, 
                              row_heights=[0.7, 0.3],
                              subplot_titles=("Price & Signals", "Buy/Sell Confidence"))
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=signals_df.index,
                    y=signals_df['close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='royalblue')
                ),
                row=1, col=1
            )
            
            # Add buy signals
            buy_signals = signals_df[signals_df['signal'] == 'BUY']
            if not buy_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        name='Buy Signal',
                        marker=dict(color='green', size=10, symbol='triangle-up')
                    ),
                    row=1, col=1
                )
            
            # Add sell signals
            sell_signals = signals_df[signals_df['signal'] == 'SELL']
            if not sell_signals.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        name='Sell Signal',
                        marker=dict(color='red', size=10, symbol='triangle-down')
                    ),
                    row=1, col=1
                )
            
            # Add confidence
            if 'confidence' in signals_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=signals_df.index,
                        y=signals_df['confidence'],
                        name='Confidence',
                        marker=dict(
                            color=np.where(signals_df['signal'] == 'BUY', 'green', 
                                        np.where(signals_df['signal'] == 'SELL', 'red', 'gray'))
                        )
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Chart with Trading Signals",
                height=600,
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Confidence", row=2, col=1)
            
            # Show plot
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error in plot_price_chart: {e}")
            st.code(traceback.format_exc())

    def display_backtest_results(backtest):
        try:
            if 'error' in backtest:
                st.error(backtest['error'])
                return
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Initial Capital", f"${backtest['initial_capital']:.2f}")
            with col2:
                st.metric("Final Value", f"${backtest['final_value']:.2f}")
            with col3:
                st.metric("Total Return", f"{backtest['total_return']:.2f}%")
            with col4:
                st.metric("Total Trades", backtest['total_trades'])
            
            # Plot portfolio value
            if 'signals_df' in backtest and 'portfolio_value' in backtest['signals_df'].columns:
                st.subheader("Portfolio Value Over Time")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=backtest['signals_df'].index,
                        y=backtest['signals_df']['portfolio_value'],
                        mode='lines',
                        name='Portfolio Value',
                        line=dict(color='green')
                    )
                )
                fig.update_layout(
                    height=400,
                    hovermode="x unified",
                    yaxis=dict(title="Portfolio Value ($)")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Display trades
            st.subheader("Trade History")
            if backtest['trades']:
                trades_df = pd.DataFrame(backtest['trades'])
                st.dataframe(trades_df)
            else:
                st.info("No trades were executed")
        except Exception as e:
            st.error(f"Error in display_backtest_results: {e}")
            st.code(traceback.format_exc())

    # Create prediction system with caching
    @st.cache_resource
    def get_prediction_system():
        return PredictionSystem(models_dir='models')
    
    try:
        system = get_prediction_system()
        st.success("✅ Prediction system initialized successfully")
    
        # Create the Streamlit app
        st.markdown("### Predict stock, forex, and crypto price movements with machine learning")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Predict", "Backtest"])
        
        # Predict tab
        with tab1:
            st.header("Price Prediction")
            
            # Input form for prediction
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                asset_type = st.selectbox("Asset Type", ["stock", "forex", "crypto"], help="Select the type of asset")
            with col2:
                if asset_type == "stock":
                    symbol = st.text_input("Symbol", "AAPL", help="Enter stock symbol (e.g., AAPL, MSFT)")
                elif asset_type == "forex":
                    symbol = st.text_input("Symbol", "EURUSD", help="Enter forex pair (e.g., EURUSD, GBPUSD)")
                else:
                    symbol = st.text_input("Symbol", "BTC-USD", help="Enter crypto symbol (e.g., BTC-USD, ETH-USD)")
            with col3:
                period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=2, help="Period of historical data to load")
            with col4:
                interval = st.selectbox("Interval", ["1d" , "1h" , "1wk"], index=0, help="Data interval")
            
            col1, col2 = st.columns(2)
            with col1:
                predict_button = st.button("Get Prediction", type="primary", use_container_width=True)
            with col2:
                train_button = st.button("Train New Model", type="secondary", use_container_width=True)
            
            if train_button:
                with st.spinner("Training models... This may take a few minutes..."):
                    try:
                        # FIXED: Use selected period instead of hardcoded "2y"
                        training_period = period if period in ["2y", "5y", "max"] else "2y"  # Use longer period for training
                        success = system.train_models(symbol, asset_type, training_period, interval)
                        if success:
                            st.success(f"Models for {symbol} trained successfully using {training_period} of data!")
                        else:
                            st.error(f"Failed to train models for {symbol}")
                    except Exception as e:
                        st.error(f"Error in training models: {e}")
                        st.code(traceback.format_exc())
            
            if predict_button or train_button:
                with st.spinner("Getting prediction..."):
                    try:
                        prediction = system.get_prediction(symbol, asset_type, period, interval)
                        display_prediction(prediction)
                        plot_price_chart(symbol, asset_type, period, interval)
                    except Exception as e:
                        st.error(f"Error in getting prediction: {e}")
                        st.code(traceback.format_exc())
        
        # Backtest tab
        with tab2:
            st.header("Strategy Backtesting")
            
            # Input form for backtesting
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                backtest_asset_type = st.selectbox("Asset Type", ["stock", "forex", "crypto"], key="bt_asset_type")
            with col2:
                if backtest_asset_type == "stock":
                    backtest_symbol = st.text_input("Symbol", "AAPL", key="bt_symbol")
                elif backtest_asset_type == "forex":
                    backtest_symbol = st.text_input("Symbol", "EURUSD", key="bt_symbol")
                else:
                    backtest_symbol = st.text_input("Symbol", "BTC-USD", key="bt_symbol")
            with col3:
                backtest_period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=1, key="bt_period")
            with col4:
                backtest_interval = st.selectbox("Interval", ["1d"], index=0, key="bt_interval")
            
            backtest_button = st.button("Run Backtest", type="primary")
            
            if backtest_button:
                with st.spinner("Running backtest..."):
                    try:
                        # Make sure models are loaded or trained
                        if system.model_trainer.direction_model is None or system.model_trainer.price_model is None:
                            st.info(f"Models for {backtest_symbol} not loaded. Loading or training now...")
                            if not system.load_models(backtest_symbol):
                                st.info(f"Training new models for {backtest_symbol}...")
                                # FIXED: Use longer period for training
                                training_period = "2y" if backtest_period in ["6mo", "1y"] else backtest_period
                                system.train_models(backtest_symbol, backtest_asset_type, training_period, backtest_interval)
                        
                        # Run backtest
                        backtest = system.backtest_strategy(backtest_symbol, backtest_asset_type, backtest_period, backtest_interval)
                        display_backtest_results(backtest)
                    except Exception as e:
                        st.error(f"Error in backtesting: {e}")
                        st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"Error initializing app components: {e}")
        st.code(traceback.format_exc())
        
except Exception as main_error:
    st.error(f"Critical error: {main_error}")
    st.code(traceback.format_exc())