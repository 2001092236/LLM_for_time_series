"""
Time Series Feature Extractor
Extracts statistical features for LLM analysis
"""

import numpy as np
from typing import Optional, Dict, Any
from scipy import stats
from scipy.signal import find_peaks


class TSFeatureExtractor:
    """Extracts comprehensive features from time series for LLM analysis"""

    def __init__(self, y: np.ndarray, x: Optional[np.ndarray] = None, freq: Optional[int] = None):
        """
        Args:
            y: Time series values
            x: Time points (optional)
            freq: Frequency/period for seasonal analysis (optional)
        """
        self.y = np.array(y).flatten()
        self.x = x if x is not None else np.arange(len(y))
        self.freq = freq
        self._validate()

    def _validate(self):
        """Validate input data"""
        if len(self.y) < 10:
            raise ValueError("Time series too short (minimum 10 points)")
        if np.any(np.isnan(self.y)):
            # Fill NaN with interpolation
            mask = np.isnan(self.y)
            self.y[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                self.y[~mask]
            )

    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all features and return as dictionary
        Designed for LLM consumption
        """
        features = {}

        # Basic statistics
        features.update(self._basic_stats())

        # Trend analysis
        features.update(self._trend_analysis())

        # Seasonality analysis
        features.update(self._seasonality_analysis())

        # Stationarity
        features.update(self._stationarity_tests())

        # Autocorrelation
        features.update(self._autocorrelation_features())

        # Distribution shape
        features.update(self._distribution_features())

        # Peaks and patterns
        features.update(self._pattern_features())

        # Complexity measures
        features.update(self._complexity_features())

        return features

    def _basic_stats(self) -> Dict[str, float]:
        """Basic statistical measures"""
        return {
            'length': int(len(self.y)),
            'mean': float(np.mean(self.y)),
            'std': float(np.std(self.y)),
            'min': float(np.min(self.y)),
            'max': float(np.max(self.y)),
            'range': float(np.ptp(self.y)),
            'median': float(np.median(self.y)),
            'variance': float(np.var(self.y)),
            'coef_of_variation': float(np.std(self.y) / np.abs(np.mean(self.y))) if np.mean(self.y) != 0 else 0.0,
        }

    def _trend_analysis(self) -> Dict[str, float]:
        """Analyze trend component"""
        n = len(self.y)
        t = np.arange(n)

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, self.y)

        # Trend strength (1 - var(residual)/var(original))
        trend_line = slope * t + intercept
        residual = self.y - trend_line
        trend_strength = max(0, 1 - np.var(residual) / np.var(self.y)) if np.var(self.y) > 0 else 0

        # Trend direction
        if abs(slope) < 1e-10:
            trend_direction = "stationary"
        elif slope > 0:
            trend_direction = "upward"
        else:
            trend_direction = "downward"

        return {
            'trend_slope': float(slope),
            'trend_intercept': float(intercept),
            'trend_r_squared': float(r_value ** 2),
            'trend_p_value': float(p_value),
            'trend_strength': float(trend_strength),
            'trend_direction': trend_direction,
        }

    def _seasonality_analysis(self) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        n = len(self.y)

        # Estimate period using autocorrelation if not provided
        if self.freq is None:
            # Use FFT to find dominant frequency
            fft = np.fft.fft(self.y - np.mean(self.y))
            freqs = np.fft.fftfreq(n)
            positive_freqs = freqs[:n//2]
            magnitudes = np.abs(fft[:n//2])

            # Find peak frequency (excluding DC component)
            if len(magnitudes) > 1:
                peak_idx = np.argmax(magnitudes[1:]) + 1
                dominant_freq = positive_freqs[peak_idx]
                estimated_period = int(1 / dominant_freq) if dominant_freq > 0 else n
                estimated_period = min(estimated_period, n // 2)
            else:
                estimated_period = n // 4
        else:
            estimated_period = self.freq

        # Compute seasonal strength
        if estimated_period > 1 and estimated_period < n // 2:
            seasonal = np.zeros(n)
            for i in range(estimated_period):
                indices = np.arange(i, n, estimated_period)
                seasonal[indices] = np.mean(self.y[indices])
            seasonal_strength = float(np.var(seasonal) / np.var(self.y)) if np.var(self.y) > 0 else 0
        else:
            seasonal_strength = 0.0

        return {
            'estimated_period': int(estimated_period),
            'seasonal_strength': float(min(1.0, seasonal_strength)),
            'has_seasonality': seasonal_strength > 0.1,
        }

    def _stationarity_tests(self) -> Dict[str, Any]:
        """Test for stationarity using ADF test"""
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(self.y, autolag='AIC')
            adf_statistic = result[0]
            p_value = result[1]
            is_stationary = p_value < 0.05

            return {
                'adf_statistic': float(adf_statistic),
                'adf_pvalue': float(p_value),
                'is_stationary': is_stationary,
                'stationarity_interpretation': "stationary" if is_stationary else "non-stationary",
            }
        except ImportError:
            # Fallback: simple variance ratio test
            n = len(self.y)
            first_half_var = np.var(self.y[:n//2])
            second_half_var = np.var(self.y[n//2:])
            variance_ratio = first_half_var / second_half_var if second_half_var > 0 else 1

            return {
                'adf_statistic': None,
                'adf_pvalue': None,
                'is_stationary': 0.5 < variance_ratio < 2.0,
                'stationarity_interpretation': "approximately stationary" if 0.5 < variance_ratio < 2.0 else "non-stationary",
            }

    def _autocorrelation_features(self) -> Dict[str, float]:
        """Autocorrelation analysis"""
        try:
            from statsmodels.tsa.stattools import acf, pacf

            max_lag = min(40, len(self.y) // 4)
            acf_values = acf(self.y, nlags=max_lag, fft=True)
            pacf_values = pacf(self.y, nlags=min(max_lag, len(self.y) // 2 - 1))

            # Find first lag where ACF drops below significance
            significance = 1.96 / np.sqrt(len(self.y))
            first_insignificant = next(
                (i for i, v in enumerate(acf_values[1:], 1) if abs(v) < significance),
                max_lag
            )

            return {
                'acf_lag1': float(acf_values[1]) if len(acf_values) > 1 else 0.0,
                'acf_lag2': float(acf_values[2]) if len(acf_values) > 2 else 0.0,
                'pacf_lag1': float(pacf_values[1]) if len(pacf_values) > 1 else 0.0,
                'pacf_lag2': float(pacf_values[2]) if len(pacf_values) > 2 else 0.0,
                'acf_decay_lag': int(first_insignificant),
                'has_strong_autocorr': abs(acf_values[1]) > 0.5 if len(acf_values) > 1 else False,
            }
        except ImportError:
            # Simple autocorrelation calculation
            y_centered = self.y - np.mean(self.y)
            n = len(y_centered)
            acf_lag1 = np.correlate(y_centered[:-1], y_centered[1:])[0] / (np.var(self.y) * (n - 1))

            return {
                'acf_lag1': float(acf_lag1),
                'acf_lag2': 0.0,
                'pacf_lag1': float(acf_lag1),
                'pacf_lag2': 0.0,
                'acf_decay_lag': 1,
                'has_strong_autocorr': abs(acf_lag1) > 0.5,
            }

    def _distribution_features(self) -> Dict[str, float]:
        """Distribution shape features"""
        return {
            'skewness': float(stats.skew(self.y)),
            'kurtosis': float(stats.kurtosis(self.y)),
            'is_normal': float(stats.normaltest(self.y)[1]) > 0.05 if len(self.y) >= 20 else None,
            'percentile_25': float(np.percentile(self.y, 25)),
            'percentile_75': float(np.percentile(self.y, 75)),
            'iqr': float(np.percentile(self.y, 75) - np.percentile(self.y, 25)),
        }

    def _pattern_features(self) -> Dict[str, Any]:
        """Detect peaks and patterns"""
        # Find peaks
        peaks, properties = find_peaks(self.y, prominence=np.std(self.y) * 0.5)
        troughs, _ = find_peaks(-self.y, prominence=np.std(self.y) * 0.5)

        # Zero crossings (around mean)
        mean_centered = self.y - np.mean(self.y)
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)

        return {
            'num_peaks': int(len(peaks)),
            'num_troughs': int(len(troughs)),
            'zero_crossings': int(zero_crossings),
            'peaks_per_100_points': float(len(peaks) / len(self.y) * 100),
        }

    def _complexity_features(self) -> Dict[str, float]:
        """Measure time series complexity"""
        # Approximate entropy (simplified)
        diff = np.diff(self.y)
        turning_points = np.sum(diff[:-1] * diff[1:] < 0)
        turning_point_rate = turning_points / (len(self.y) - 2)

        # First difference statistics
        first_diff_std = np.std(diff)

        # Hurst exponent approximation (R/S analysis simplified)
        n = len(self.y)
        mean = np.mean(self.y)
        cumdev = np.cumsum(self.y - mean)
        r = np.max(cumdev) - np.min(cumdev)
        s = np.std(self.y)
        rs = r / s if s > 0 else 0
        hurst_approx = np.log(rs) / np.log(n) if rs > 0 and n > 1 else 0.5

        return {
            'turning_point_rate': float(turning_point_rate),
            'first_diff_std': float(first_diff_std),
            'hurst_exponent_approx': float(hurst_approx),
            'is_mean_reverting': hurst_approx < 0.5,
            'is_trending': hurst_approx > 0.5,
        }

    def get_summary(self) -> str:
        """Generate human-readable summary for LLM"""
        features = self.extract_all()

        summary = f"""Time Series Analysis Summary:

Length: {features['length']} points
Mean: {features['mean']:.4f}, Std: {features['std']:.4f}
Range: [{features['min']:.4f}, {features['max']:.4f}]

Trend: {features['trend_direction']} (strength={features['trend_strength']:.2f}, slope={features['trend_slope']:.6f})
Seasonality: {"Yes" if features['has_seasonality'] else "No"} (period~{features['estimated_period']}, strength={features['seasonal_strength']:.2f})
Stationarity: {features['stationarity_interpretation']} (ADF p-value={features.get('adf_pvalue', 'N/A')})

Autocorrelation: ACF(1)={features['acf_lag1']:.3f}, {"Strong" if features['has_strong_autocorr'] else "Weak"}
Distribution: skewness={features['skewness']:.3f}, kurtosis={features['kurtosis']:.3f}
Complexity: Hurst≈{features['hurst_exponent_approx']:.3f} ({"mean-reverting" if features['is_mean_reverting'] else "trending"})
"""
        return summary


# Quick test
if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 500)
    y = 5 * np.sin(t) + 0.02 * t + np.random.normal(0, 0.5, 500)

    extractor = TSFeatureExtractor(y, t)
    features = extractor.extract_all()

    print("Extracted Features:")
    print("-" * 50)
    for key, value in features.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print(extractor.get_summary())
