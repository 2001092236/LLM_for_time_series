"""
Data Generator for Time Series Analysis
Generates synthetic, physics-based, and real-world time series data
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class TimeSeriesData:
    """Container for time series data"""
    x: np.ndarray
    y: np.ndarray
    description: str
    context: str = ""

    def to_dict(self) -> dict:
        return {
            'x': self.x,
            'y': self.y,
            'description': self.description,
            'context': self.context
        }


class DataGenerator:
    """Generator for all types of time series data"""

    # ==================== SYNTHETIC DATA ====================

    @staticmethod
    def synthetic_sine(
        n_points: int = 1000,
        noise_std: float = 0.5,
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Composition of sine waves with noise
        y = 5*sin(t) + 3*sin(2t) + 2*sin(3t) + noise
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.linspace(0, 4 * np.pi, n_points)
        y = (5 * np.sin(t) +
             3 * np.sin(2 * t) +
             2 * np.sin(3 * t) +
             np.random.normal(0, noise_std, n_points))

        return TimeSeriesData(
            x=t,
            y=y,
            description="Synthetic sine composition",
            context="Superposition of three sine waves (frequencies 1, 2, 3) with Gaussian noise. "
                    "Represents periodic signals common in signal processing."
        )

    @staticmethod
    def synthetic_trend_seasonal(
        n_points: int = 500,
        trend_slope: float = 0.02,
        seasonal_period: int = 50,
        noise_std: float = 0.3,
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Time series with trend + seasonality + noise
        Good for testing decomposition methods
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.arange(n_points)
        trend = trend_slope * t
        seasonal = 2 * np.sin(2 * np.pi * t / seasonal_period)
        noise = np.random.normal(0, noise_std, n_points)
        y = trend + seasonal + noise

        return TimeSeriesData(
            x=t,
            y=y,
            description="Trend + Seasonal + Noise",
            context=f"Linear trend (slope={trend_slope}) with seasonal component "
                    f"(period={seasonal_period}) and Gaussian noise."
        )

    # ==================== PHYSICS-BASED DATA ====================

    @staticmethod
    def physics_damped_oscillator(
        n_points: int = 1000,
        zeta: float = 0.1,
        omega: float = 2 * np.pi,
        noise_std: float = 0.02,
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Damped harmonic oscillator (pendulum with friction)
        Solution: y(t) = A * exp(-zeta*omega*t) * sin(omega_d*t)
        where omega_d = omega * sqrt(1 - zeta^2)

        Physical equation: d²y/dt² + 2*zeta*omega*(dy/dt) + omega²*y = 0
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.linspace(0, 10, n_points)
        omega_d = omega * np.sqrt(1 - zeta**2)  # damped frequency
        y = np.exp(-zeta * omega * t) * np.sin(omega_d * t)
        y += np.random.normal(0, noise_std, len(y))

        return TimeSeriesData(
            x=t,
            y=y,
            description="Damped harmonic oscillator",
            context=f"Physics: damped pendulum. Equation: d²y/dt² + 2ζω(dy/dt) + ω²y = 0. "
                    f"Parameters: ζ={zeta} (damping ratio), ω={omega:.2f} rad/s. "
                    f"Underdamped system with exponential decay envelope."
        )

    @staticmethod
    def physics_lorenz_attractor(
        n_points: int = 5000,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8/3,
        dt: float = 0.01,
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Lorenz attractor - chaotic system
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

        Returns x-component of the trajectory
        """
        if seed is not None:
            np.random.seed(seed)

        # Initial conditions
        x, y, z = 1.0, 1.0, 1.0
        xs = [x]

        for _ in range(n_points - 1):
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt
            x, y, z = x + dx, y + dy, z + dz
            xs.append(x)

        t = np.arange(n_points) * dt
        y_out = np.array(xs)

        return TimeSeriesData(
            x=t,
            y=y_out,
            description="Lorenz attractor (chaotic)",
            context=f"Lorenz system - deterministic chaos. "
                    f"Parameters: σ={sigma}, ρ={rho}, β={beta:.2f}. "
                    f"Exhibits sensitive dependence on initial conditions."
        )

    @staticmethod
    def physics_radioactive_decay(
        n_points: int = 500,
        half_life: float = 50.0,
        initial_count: float = 1000.0,
        noise_std: float = 10.0,
        seed: Optional[int] = None
    ) -> TimeSeriesData:
        """
        Radioactive decay: N(t) = N0 * exp(-lambda*t)
        where lambda = ln(2) / half_life
        """
        if seed is not None:
            np.random.seed(seed)

        t = np.linspace(0, 5 * half_life, n_points)
        decay_constant = np.log(2) / half_life
        y = initial_count * np.exp(-decay_constant * t)
        y += np.random.normal(0, noise_std, len(y))
        y = np.maximum(y, 0)  # counts can't be negative

        return TimeSeriesData(
            x=t,
            y=y,
            description="Radioactive decay",
            context=f"Exponential decay process. N(t) = N₀·exp(-λt). "
                    f"Half-life T½={half_life}, λ=ln(2)/T½={decay_constant:.4f}. "
                    f"Initial count N₀={initial_count}."
        )

    # ==================== REAL-WORLD DATA ====================

    @staticmethod
    def real_stock_prices(
        ticker: str = 'AAPL',
        period: str = '1y'
    ) -> TimeSeriesData:
        """
        Real stock prices from Yahoo Finance
        Requires: pip install yfinance
        """
        try:
            import yfinance as yf
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            y = df['Close'].values.flatten()
            x = np.arange(len(y))

            return TimeSeriesData(
                x=x,
                y=y,
                description=f"Stock prices: {ticker}",
                context=f"Daily closing prices for {ticker} stock. "
                        f"Period: {period}. Financial time series with "
                        f"potential trends, volatility clustering, and non-stationarity."
            )
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch stock data: {e}")

    @staticmethod
    def real_co2_data() -> TimeSeriesData:
        """
        Mauna Loa CO2 concentration data from statsmodels
        Classic dataset with trend and seasonality
        """
        try:
            from statsmodels.datasets import co2
            data = co2.load_pandas().data

            # Handle missing values
            data = data.ffill().dropna()

            y = data['co2'].values
            x = np.arange(len(y))

            return TimeSeriesData(
                x=x,
                y=y,
                description="Mauna Loa CO2 concentration",
                context="Weekly CO2 concentration (ppm) at Mauna Loa Observatory. "
                        "Shows clear upward trend (climate change) and annual seasonality "
                        "(vegetation cycle). Classic time series dataset."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CO2 data: {e}")

    @staticmethod
    def real_airline_passengers() -> TimeSeriesData:
        """
        Classic Box-Jenkins airline passengers dataset
        Monthly totals of international airline passengers (1949-1960)
        """
        try:
            # Classic dataset - monthly airline passengers
            passengers = np.array([
                112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118,
                115, 126, 141, 135, 125, 149, 170, 170, 158, 133, 114, 140,
                145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166,
                171, 180, 193, 181, 183, 218, 230, 242, 209, 191, 172, 194,
                196, 196, 236, 235, 229, 243, 264, 272, 237, 211, 180, 201,
                204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229,
                242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
                284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306,
                315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336,
                340, 318, 362, 348, 363, 435, 491, 505, 404, 359, 310, 337,
                360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405,
                417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432
            ], dtype=float)

            x = np.arange(len(passengers))

            return TimeSeriesData(
                x=x,
                y=passengers,
                description="Airline passengers (1949-1960)",
                context="Monthly totals of international airline passengers (thousands). "
                        "Classic Box-Jenkins dataset. Shows multiplicative seasonality "
                        "and exponential trend. Period 1949-1960."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load airline data: {e}")

    # ==================== UTILITY METHODS ====================

    @classmethod
    def get_all_datasets(cls) -> dict:
        """Returns dictionary of all available dataset generators"""
        return {
            # Synthetic
            'synthetic_sine': cls.synthetic_sine,
            'synthetic_trend_seasonal': cls.synthetic_trend_seasonal,
            # Physics
            'physics_damped_oscillator': cls.physics_damped_oscillator,
            'physics_lorenz': cls.physics_lorenz_attractor,
            'physics_radioactive_decay': cls.physics_radioactive_decay,
            # Real
            'real_stock': cls.real_stock_prices,
            'real_co2': cls.real_co2_data,
            'real_airline': cls.real_airline_passengers,
        }

    @classmethod
    def load(cls, name: str, **kwargs) -> TimeSeriesData:
        """
        Load dataset by name

        Args:
            name: Dataset name (see get_all_datasets() for options)
            **kwargs: Parameters for the generator

        Returns:
            TimeSeriesData object
        """
        datasets = cls.get_all_datasets()
        if name not in datasets:
            available = ', '.join(datasets.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")

        return datasets[name](**kwargs)


# Quick test
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    gen = DataGenerator()

    # Test all datasets
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()

    datasets = [
        ('synthetic_sine', {}),
        ('synthetic_trend_seasonal', {}),
        ('physics_damped_oscillator', {}),
        ('physics_lorenz', {}),
        ('physics_radioactive_decay', {}),
        ('real_airline', {}),
        ('real_co2', {}),
    ]

    for ax, (name, kwargs) in zip(axes, datasets):
        try:
            data = gen.load(name, **kwargs)
            ax.plot(data.x, data.y, linewidth=0.8)
            ax.set_title(data.description)
            ax.grid(True, alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax.set_title(name)

    # Hide empty subplots
    for ax in axes[len(datasets):]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('outputs/visualizations/data_samples.png', dpi=150)
    plt.show()

    print("\nAvailable datasets:")
    for name in gen.get_all_datasets().keys():
        print(f"  - {name}")
