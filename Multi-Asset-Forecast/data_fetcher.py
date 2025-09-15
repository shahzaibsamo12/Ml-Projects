import yfinance as yf
import pandas as pd
import ccxt
import datetime
import time

class DataFetcher:
    def __init__(self):
        self.binance = ccxt.binance()
    
    def get_stock_data(self, ticker, period="1y", interval="1d"):
        """
        Fetch stock data from Yahoo Finance
        
        Parameters:
        ticker (str): Stock symbol (e.g., 'AAPL', 'MSFT')
        period (str): Period to fetch (e.g., '1d', '5d', '1mo', '3mo', '1y', '5y', 'max')
        interval (str): Data interval (e.g., '1m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo')
        
        Returns:
        pandas.DataFrame: Historical data
        """
        try:
            data = yf.download(ticker, period=period, interval=interval)
            if data.empty:
                print(f"No data found for {ticker}")
                return None
            
            # Ensure column names are strings, not tuples (happens with some yfinance data)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Convert all column names to lowercase for consistency
            data.columns = [col.lower() for col in data.columns]
            
            # Add symbol column
            data['symbol'] = ticker
            
            # Format Date as string if it's datetime
            if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            return data
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return None
    
    def get_forex_data(self, pair, period="1y", interval="1d"):
        """
        Fetch forex data from Yahoo Finance
        
        Parameters:
        pair (str): Forex pair (e.g., 'EURUSD=X', 'GBPUSD=X')
        period (str): Period to fetch
        interval (str): Data interval
        
        Returns:
        pandas.DataFrame: Historical data
        """
        # Add =X to forex pair if not already added
        if not pair.endswith('=X'):
            pair = f"{pair}=X"
            
        data = self.get_stock_data(pair, period, interval)
        
        # Make sure the data has the required columns for technical indicators
        if data is not None and 'volume' not in data.columns:
            # Some forex data doesn't have volume, add a dummy volume column
            data['volume'] = 1000  # Add dummy value
            print(f"Added dummy volume column to forex data for {pair}")
            
        return data

    def get_crypto_data(self, symbol, period="1y", interval="1d"):
        """
        Fetch cryptocurrency data from Yahoo Finance
        
        Parameters:
        symbol (str): Crypto symbol (e.g., 'BTC-USD', 'ETH-USD')
        period (str): Period to fetch
        interval (str): Data interval
        
        Returns:
        pandas.DataFrame: Historical data
        """
        data = self.get_stock_data(symbol, period, interval)
        
        # Make sure the data has the required columns for technical indicators
        if data is not None and 'volume' not in data.columns:
            # Some crypto data doesn't have volume, add a dummy volume column
            data['volume'] = 1000  # Add dummy value
            print(f"Added dummy volume column to crypto data for {symbol}")
            
        return data
    
    def get_binance_crypto_data(self, symbol, timeframe='1d', limit=365):
        """
        Fetch cryptocurrency data from Binance
        
        Parameters:
        symbol (str): Crypto pair (e.g., 'BTC/USDT')
        timeframe (str): Timeframe (e.g., '1m', '5m', '1h', '1d')
        limit (int): Number of candles to fetch
        
        Returns:
        pandas.DataFrame: Historical data
        """
        try:
            # Fetch OHLCV data
            ohlcv = self.binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Add symbol column
            df['symbol'] = symbol
            
            return df
        except Exception as e:
            print(f"Error fetching Binance data for {symbol}: {e}")
            return None
        

# Test function
if __name__ == "__main__":
    fetcher = DataFetcher()
    
    # Test stock data
    apple_data = fetcher.get_stock_data("AAPL", period="1mo")
    print("\nStock Data Sample:")
    print(apple_data.head() if apple_data is not None else "No data")
    
    # Test forex data
    eurusd_data = fetcher.get_forex_data("EURUSD", period="1mo")
    print("\nForex Data Sample:")
    print(eurusd_data.head() if eurusd_data is not None else "No data")
    
    # Test crypto data
    btc_data = fetcher.get_crypto_data("BTC-USD", period="1mo")
    print("\nCrypto Data Sample:")
    print(btc_data.head() if btc_data is not None else "No data")
    
    # Test Binance data
    binance_btc = fetcher.get_binance_crypto_data("BTC/USDT", timeframe="1d", limit=30)
    print("\nBinance Data Sample:")
    print(binance_btc.head() if binance_btc is not None else "No data")