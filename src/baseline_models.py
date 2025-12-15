"""
Baseline Models for Time Series Analysis
Provides quick baseline forecasts for comparison
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

warnings.filterwarnings('ignore')


@dataclass
class ModelResult:
    """Container for model results"""
    name: str
    model: Any
    fitted_values: np.ndarray
    forecast: np.ndarray
    metrics: Dict[str, float]
    success: bool
    error_message: Optional[str] = None


class BaselineModels:
    """
    Run baseline models for time series comparison
    Supports: ARIMA, Prophet, AutoReg, Exponential Smoothing, GPR
    """

    def __init__(self, y: np.ndarray, x: Optional[np.ndarray] = None, train_ratio: float = 0.8):
        """
        Args:
            y: Time series values
            x: Time points (optional)
            train_ratio: Fraction for train/test split
        """
        self.y = np.array(y).flatten()
        self.x = x if x is not None else np.arange(len(y))
        self.train_ratio = train_ratio

        # Train/test split
        self.split_idx = int(len(self.y) * train_ratio)
        self.y_train = self.y[:self.split_idx]
        self.y_test = self.y[self.split_idx:]
        self.x_train = self.x[:self.split_idx]
        self.x_test = self.x[self.split_idx:]

        self.results: Dict[str, ModelResult] = {}

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute forecast metrics"""
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        else:
            mape = float('inf')

        # R-squared
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
        }

    # ==================== ARIMA ====================

    def run_arima(self, order: Tuple[int, int, int] = (1, 1, 1)) -> ModelResult:
        """
        ARIMA model (AutoRegressive Integrated Moving Average)
        Good for non-stationary data with trend
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            model = ARIMA(self.y_train, order=order)
            fitted = model.fit()

            # Fitted values
            fitted_values = fitted.fittedvalues

            # Forecast
            forecast = fitted.forecast(steps=len(self.y_test))

            metrics = self._compute_metrics(self.y_test, forecast)
            metrics['aic'] = float(fitted.aic)
            metrics['bic'] = float(fitted.bic)

            result = ModelResult(
                name=f'ARIMA{order}',
                model=fitted,
                fitted_values=np.array(fitted_values),
                forecast=np.array(forecast),
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name=f'ARIMA{order}',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['arima'] = result
        return result

    # ==================== AUTO ARIMA ====================

    def run_auto_arima(self, max_p: int = 3, max_q: int = 3, max_d: int = 2) -> ModelResult:
        """
        Automatic ARIMA with order selection based on AIC
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA

            best_aic = float('inf')
            best_order = (1, 1, 1)
            best_model = None

            for p in range(max_p + 1):
                for d in range(max_d + 1):
                    for q in range(max_q + 1):
                        if p == 0 and q == 0:
                            continue
                        try:
                            model = ARIMA(self.y_train, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                        except:
                            continue

            if best_model is None:
                raise ValueError("No valid ARIMA model found")

            forecast = best_model.forecast(steps=len(self.y_test))
            metrics = self._compute_metrics(self.y_test, forecast)
            metrics['aic'] = float(best_model.aic)
            metrics['bic'] = float(best_model.bic)
            metrics['selected_order'] = str(best_order)

            result = ModelResult(
                name=f'AutoARIMA{best_order}',
                model=best_model,
                fitted_values=np.array(best_model.fittedvalues),
                forecast=np.array(forecast),
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name='AutoARIMA',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['auto_arima'] = result
        return result

    # ==================== PROPHET ====================

    def run_prophet(self, yearly_seasonality: bool = False, weekly_seasonality: bool = False) -> ModelResult:
        """
        Facebook Prophet model
        Good for data with multiple seasonalities
        """
        try:
            from prophet import Prophet

            # Prepare dataframe for Prophet
            df_train = pd.DataFrame({
                'ds': pd.date_range(start='2020-01-01', periods=len(self.y_train), freq='D'),
                'y': self.y_train
            })

            model = Prophet(
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False
            )
            model.fit(df_train)

            # Create future dataframe
            future = model.make_future_dataframe(periods=len(self.y_test))
            prediction = model.predict(future)

            fitted_values = prediction['yhat'].values[:len(self.y_train)]
            forecast = prediction['yhat'].values[len(self.y_train):]

            metrics = self._compute_metrics(self.y_test, forecast)

            result = ModelResult(
                name='Prophet',
                model=model,
                fitted_values=fitted_values,
                forecast=forecast,
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name='Prophet',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['prophet'] = result
        return result

    # ==================== AutoReg ====================

    def run_autoreg(self, lags: int = 10) -> ModelResult:
        """
        AutoRegressive model
        Simple AR model for stationary data
        """
        try:
            from statsmodels.tsa.ar_model import AutoReg

            model = AutoReg(self.y_train, lags=lags)
            fitted = model.fit()

            # Fitted values
            fitted_values = fitted.fittedvalues

            # Forecast
            forecast = fitted.predict(
                start=len(self.y_train),
                end=len(self.y_train) + len(self.y_test) - 1
            )

            metrics = self._compute_metrics(self.y_test, forecast)

            result = ModelResult(
                name=f'AR({lags})',
                model=fitted,
                fitted_values=np.array(fitted_values),
                forecast=np.array(forecast),
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name=f'AR({lags})',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['autoreg'] = result
        return result

    # ==================== Exponential Smoothing ====================

    def run_exp_smoothing(self, seasonal_periods: Optional[int] = None) -> ModelResult:
        """
        Exponential Smoothing (Holt-Winters)
        Good for data with trend and seasonality
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            if seasonal_periods and seasonal_periods > 1:
                model = ExponentialSmoothing(
                    self.y_train,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(
                    self.y_train,
                    trend='add',
                    seasonal=None
                )

            fitted = model.fit()
            fitted_values = fitted.fittedvalues
            forecast = fitted.forecast(len(self.y_test))

            metrics = self._compute_metrics(self.y_test, forecast)
            metrics['aic'] = float(fitted.aic)

            result = ModelResult(
                name='ExponentialSmoothing',
                model=fitted,
                fitted_values=np.array(fitted_values),
                forecast=np.array(forecast),
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name='ExponentialSmoothing',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['exp_smoothing'] = result
        return result

    # ==================== Gaussian Process Regression ====================

    def run_gpr(self) -> ModelResult:
        """
        Gaussian Process Regression
        Good for uncertainty estimation
        """
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

            # Reshape for sklearn
            X_train = self.x_train.reshape(-1, 1)
            X_test = self.x_test.reshape(-1, 1)

            # Define kernel
            kernel = ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)

            model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
            model.fit(X_train, self.y_train)

            fitted_values, _ = model.predict(X_train, return_std=True)
            forecast, forecast_std = model.predict(X_test, return_std=True)

            metrics = self._compute_metrics(self.y_test, forecast)
            metrics['mean_uncertainty'] = float(np.mean(forecast_std))

            result = ModelResult(
                name='GPR',
                model=model,
                fitted_values=np.array(fitted_values),
                forecast=np.array(forecast),
                metrics=metrics,
                success=True
            )

        except Exception as e:
            result = ModelResult(
                name='GPR',
                model=None,
                fitted_values=np.array([]),
                forecast=np.array([]),
                metrics={},
                success=False,
                error_message=str(e)
            )

        self.results['gpr'] = result
        return result

    # ==================== Run All ====================

    def run_all(self) -> Dict[str, ModelResult]:
        """Run all baseline models"""
        self.run_arima()
        self.run_autoreg()
        self.run_exp_smoothing()
        self.run_gpr()

        # Prophet may be slow, run separately if needed
        try:
            self.run_prophet()
        except:
            pass

        return self.results

    def get_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of metrics for LLM analysis"""
        summary = {}
        for name, result in self.results.items():
            if result.success:
                summary[name] = {
                    'model_name': result.name,
                    'rmse': result.metrics.get('rmse', None),
                    'mae': result.metrics.get('mae', None),
                    'r2': result.metrics.get('r2', None),
                    'aic': result.metrics.get('aic', None),
                }
            else:
                summary[name] = {
                    'model_name': result.name,
                    'error': result.error_message
                }
        return summary

    def get_best_model(self) -> Optional[ModelResult]:
        """Get best model based on RMSE"""
        successful = [r for r in self.results.values() if r.success]
        if not successful:
            return None
        return min(successful, key=lambda r: r.metrics.get('rmse', float('inf')))

    def print_summary(self):
        """Print formatted summary"""
        print("\n" + "=" * 60)
        print("BASELINE MODELS COMPARISON")
        print("=" * 60)
        print(f"{'Model':<25} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
        print("-" * 60)

        for name, result in self.results.items():
            if result.success:
                rmse = result.metrics.get('rmse', 0)
                mae = result.metrics.get('mae', 0)
                r2 = result.metrics.get('r2', 0)
                print(f"{result.name:<25} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
            else:
                print(f"{result.name:<25} {'FAILED':>10} - {result.error_message[:20]}")

        best = self.get_best_model()
        if best:
            print("-" * 60)
            print(f"Best model: {best.name} (RMSE={best.metrics['rmse']:.4f})")
        print("=" * 60)


# Quick test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate test data
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    y = 5 * np.sin(t) + 0.5 * t + np.random.normal(0, 0.5, 200)

    # Run baseline models
    baseline = BaselineModels(y, t, train_ratio=0.8)
    baseline.run_all()
    baseline.print_summary()

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 6))

    # Original data
    ax.plot(t, y, 'k-', label='Original', alpha=0.7)

    # Train/test split line
    ax.axvline(x=t[baseline.split_idx], color='gray', linestyle='--', label='Train/Test Split')

    # Forecasts
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for (name, result), color in zip(baseline.results.items(), colors):
        if result.success and len(result.forecast) > 0:
            ax.plot(baseline.x_test, result.forecast, color=color,
                   label=f'{result.name} (RMSE={result.metrics["rmse"]:.2f})', linewidth=1.5)

    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Baseline Models Comparison')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/visualizations/baseline_comparison.png', dpi=150)
    plt.show()
