from scipy import signal

    # ... inside StrategyEngine class ...

    def generate_spectrogram_data(self, ticker):
        """
        Treats the stock price as an audio wave and performs Spectral Analysis.
        Returns: Frequencies, Times, and the Spectrogram Matrix (Zxx)
        """
        try:
            # 1. Get long history (2 years) for good wave resolution
            data = yf.download(ticker.split(" ")[0], period="2y", interval="1d", progress=False)
            if data.empty: return None, None, None

            # 2. Detrend the data (Remove the overall uptrend/downtrend)
            # We want to hear the 'vibration', not the direction.
            # Convert to numpy array and flatten
            prices = self.to_scalar_array(data['Close']) 
            # Calculate simple returns (differencing) to make it stationary-ish
            returns = np.diff(prices) 
            
            # 3. Apply Signal Processing (The "Sound" Analysis)
            # fs = 1.0 (Sampling frequency = 1 day)
            # nperseg = Window size (how many days to analyze at once, e.g., 60 days)
            f, t, Sxx = signal.spectrogram(returns, fs=1.0, window='hann', nperseg=60, noverlap=50)
            
            return f, t, Sxx
            
        except Exception as e:
            print(f"Spectrogram Error: {e}")
            return None, None, None

    def to_scalar_array(self, series):
        """Helper to ensure we have a clean 1D numpy array"""
        if isinstance(series, pd.DataFrame):
            return series.iloc[:, 0].to_numpy()
        return series.to_numpy()
