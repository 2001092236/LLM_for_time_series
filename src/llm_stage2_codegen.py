"""
LLM Stage 2: Code Generation and Execution
Takes the analytical report from Stage 1 and generates executable Python code
for time series processing, visualization, and forecasting.
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProcessingResult:
    """Container for processing results"""
    model: Any
    predictions: np.ndarray
    metrics: Dict[str, float]
    generated_code: str
    success: bool
    error_message: Optional[str] = None

    def plot(
        self,
        y: np.ndarray,
        x: Optional[np.ndarray] = None,
        description: str = "",
        train_ratio: float = 0.8,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot train, test, and predictions on a single figure.

        Args:
            y: Original y array (full data)
            x: Original x array (coordinates). If None, uses integer indices.
            description: Plot title description
            train_ratio: Train/test split ratio
            figsize: Figure size
            save_path: Path to save figure (optional)

        Returns:
            matplotlib Figure
        """
        if not self.success or len(self.predictions) == 0:
            print(f"Cannot plot: {self.error_message or 'No predictions'}")
            return None

        n = len(y)

        # Use provided x coordinates or integer indices as fallback
        if x is None:
            x = np.arange(n)

        # Split indices
        split_idx = int(n * train_ratio)
        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Ensure predictions match test length
        preds = self.predictions[:len(y_test)]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot data
        ax.plot(x_train, y_train, 'b-', label='Train', linewidth=1, alpha=0.8)
        ax.plot(x_test, y_test, 'g-', label='Test (actual)', linewidth=1.5)
        ax.plot(x_test[:len(preds)], preds, 'r--', label='Predictions', linewidth=2)

        # Split line
        ax.axvline(x=x[split_idx], color='gray', linestyle='--', alpha=0.5, label='Train/Test split')

        # Metrics in title
        metrics = self.metrics
        title = f"{description}\n"
        title += f"RMSE={metrics.get('rmse', 0):.4f}, "
        title += f"MAE={metrics.get('mae', 0):.4f}, "
        title += f"R²={metrics.get('r2', 0):.4f}"

        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


class LLMCodeGenerator:
    """
    Stage 2 of the LLM Agent: Generates and executes Python code
    for time series processing based on Stage 1 recommendations.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Args:
            api_key: OpenAI API key (default: from .env)
            model: Model name (default: from .env or gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY in .env")

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.generated_code = None

    def _build_system_prompt(self) -> str:
        """Build system prompt for code generation"""
        return """You are an expert Python developer specializing in time series analysis.

Generate a Python function that trains a model and returns predictions. NO VISUALIZATION.

FUNCTION SIGNATURE:
```python
def process_timeseries(x, y, description):
    # x: np.ndarray of integers [0, 1, 2, ..., n-1]
    # y: np.ndarray of float values
    # Returns: (model, predictions, metrics_dict)
```

IMPLEMENTATION STEPS:

1. TRAIN-TEST SPLIT (80/20):
```python
split_idx = int(len(y) * 0.8)
x_train, x_test = x[:split_idx], x[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
```

2. TRAIN MODEL on y_train:
   - ARIMA: model = ARIMA(y_train, order=(p,d,q)).fit()
   - AutoReg: model = AutoReg(y_train, lags=lags).fit()
   - GPR: model.fit(x_train.reshape(-1,1), y_train)
   - Prophet:
     ```python
     df_train = pd.DataFrame({
         'ds': pd.date_range(start='2020-01-01', periods=len(y_train), freq='D'),
         'y': y_train
     })
     model = Prophet(daily_seasonality=False, yearly_seasonality=False)
     model.fit(df_train)
     ```

3. PREDICT (length must equal len(y_test)):
   - ARIMA: predictions = model.forecast(steps=len(y_test))
   - AutoReg: predictions = model.predict(start=len(y_train), end=len(y_train)+len(y_test)-1)
   - GPR: predictions = model.predict(x_test.reshape(-1,1))
   - Prophet:
     ```python
     future = model.make_future_dataframe(periods=len(y_test))
     forecast = model.predict(future)
     predictions = forecast['yhat'].values[-len(y_test):]
     ```

4. COMPUTE METRICS:
```python
metrics = {
    'mse': float(mean_squared_error(y_test, predictions)),
    'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
    'mae': float(mean_absolute_error(y_test, predictions)),
    'r2': float(r2_score(y_test, predictions))
}
```

5. RETURN (no plotting!):
```python
return model, np.array(predictions).flatten(), metrics
```

RULES:
- NO plt, NO visualization, NO plotting
- predictions must be numpy array with length == len(y_test)
- Handle exceptions with try-except, fallback to simpler model if needed

USE THESE EXACT IMPORTS (copy-paste them):
```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from prophet import Prophet
```

Output ONLY the Python code. No markdown, no explanations."""

    def _build_user_prompt(
        self,
        ts_info: Dict[str, Any],
        llm_report: Dict[str, Any]
    ) -> str:
        """Build user prompt with time series info and LLM report"""

        report_json = json.dumps(llm_report, indent=2, ensure_ascii=False, default=str)

        prompt = f"""Generate `process_timeseries(x, y, description)` function.

## TIME SERIES INFO
- X: discrete integer indices [0, 1, 2, ..., {ts_info.get('length', 100)-1}] - NOT dates!
- Y shape: {ts_info.get('y_shape', 'unknown')}
- Description: {ts_info.get('description', 'Time series data')}
- Length: {ts_info.get('length', 'unknown')}
- Mean: {ts_info.get('mean', 'unknown'):.4f}
- Std: {ts_info.get('std', 'unknown'):.4f}

## LLM ANALYTICAL REPORT (Stage 1)
{report_json}

## TASK
Based on the report, generate the `process_timeseries` function that:
1. Train-test split 80/20
2. Train the recommended model: {llm_report.get('recommended_models', [{}])[0].get('name', 'ARIMA') if llm_report.get('recommended_models') else 'ARIMA'}
3. Predict on TEST SET only (predictions length == len(y_test))
4. Compute metrics: mse, rmse, mae, r2
5. Return: (model, predictions, metrics)

NO PLOTTING - just return the results.

Generate ONLY the Python function code."""

        return prompt

    def generate_code(
        self,
        ts_data: Dict[str, Any],
        llm_report: Dict[str, Any],
        temperature: float = 0.2
    ) -> str:
        """
        Generate Python code based on analytical report

        Args:
            ts_data: Dictionary with 'x', 'y', 'description' keys
            llm_report: Analytical report from Stage 1
            temperature: LLM temperature

        Returns:
            Generated Python code as string
        """
        ts_info = {
            'x_shape': ts_data['x'].shape if hasattr(ts_data['x'], 'shape') else len(ts_data['x']),
            'y_shape': ts_data['y'].shape if hasattr(ts_data['y'], 'shape') else len(ts_data['y']),
            'description': ts_data.get('description', ''),
            'length': len(ts_data['y']),
            'mean': float(np.mean(ts_data['y'])),
            'std': float(np.std(ts_data['y'])),
        }

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(ts_info, llm_report)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            max_tokens=10000
        )

        self.generated_code = response.choices[0].message.content
        return self.generated_code

    def _extract_code(self, response: str) -> str:
        """Extract Python code from markdown code blocks"""
        # Try to find ```python ... ``` blocks
        pattern = r'```python\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Try ``` ... ``` blocks
        pattern = r'```\s*(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0].strip()

        # Return as-is if no code blocks found
        return response.strip()

    def _create_execution_namespace(self) -> Dict[str, Any]:
        """Create namespace with all required imports"""
        namespace = {}

        # Execute imports
        exec("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import train_test_split
""", namespace)

        # Try importing optional libraries
        try:
            exec("from statsmodels.tsa.arima.model import ARIMA", namespace)
        except:
            pass

        try:
            exec("from statsmodels.tsa.ar_model import AutoReg", namespace)
        except:
            pass

        try:
            exec("from statsmodels.tsa.holtwinters import ExponentialSmoothing", namespace)
        except:
            pass

        try:
            exec("from prophet import Prophet", namespace)
        except:
            pass

        return namespace

    def execute_code(self, ts_data: Dict[str, Any]) -> ProcessingResult:
        """
        Execute the generated code

        Args:
            ts_data: Dictionary with 'x', 'y', 'description' keys

        Returns:
            ProcessingResult with model, predictions, and metrics
        """
        if self.generated_code is None:
            return ProcessingResult(
                model=None,
                predictions=np.array([]),
                metrics={},
                generated_code="",
                success=False,
                error_message="No code generated. Call generate_code() first."
            )

        code = self._extract_code(self.generated_code)
        namespace = self._create_execution_namespace()

        try:
            # Execute the generated code to define the function
            exec(code, namespace)

            # Check if function exists
            if 'process_timeseries' not in namespace:
                raise ValueError("Generated code does not define 'process_timeseries' function")

            # Call the function
            x = np.array(ts_data['x'])
            y = np.array(ts_data['y'])
            description = ts_data.get('description', '')

            result = namespace['process_timeseries'](x, y, description)

            # Unpack result
            if isinstance(result, tuple) and len(result) >= 3:
                model, predictions, metrics = result[0], result[1], result[2]
            else:
                raise ValueError(f"Function returned unexpected format: {type(result)}")

            # Ensure predictions is numpy array
            predictions = np.array(predictions).flatten()

            return ProcessingResult(
                model=model,
                predictions=predictions,
                metrics=metrics if isinstance(metrics, dict) else {},
                generated_code=code,
                success=True
            )

        except Exception as e:
            return ProcessingResult(
                model=None,
                predictions=np.array([]),
                metrics={},
                generated_code=code if 'code' in locals() else "",
                success=False,
                error_message=str(e)
            )

    def generate_and_execute(
        self,
        ts_data: Dict[str, Any],
        llm_report: Dict[str, Any]
    ) -> ProcessingResult:
        """
        Generate code and execute it in one step

        Args:
            ts_data: Dictionary with 'x', 'y', 'description'
            llm_report: Analytical report from Stage 1

        Returns:
            ProcessingResult
        """
        self.generate_code(ts_data, llm_report)
        return self.execute_code(ts_data)


class FallbackProcessor:
    """
    Fallback processor when LLM code generation fails.
    Implements standard processing pipeline.
    """

    @staticmethod
    def process_timeseries(
        x: np.ndarray,
        y: np.ndarray,
        description: str = "",
        model_type: str = "arima",
        show_plot: bool = True
    ) -> Tuple[Any, np.ndarray, Dict[str, float]]:
        """
        Fallback processing function

        Args:
            x: Time points (discrete indices)
            y: Values
            description: Data description
            model_type: Model to use ('arima', 'ar', 'exp_smoothing')
            show_plot: Whether to display the plot

        Returns:
            (model, predictions, metrics)
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Ensure x is discrete indices
        x = np.arange(len(y))

        # Train-test split (80/20)
        split_idx = int(len(y) * 0.8)
        x_train, x_test = x[:split_idx], x[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        model = None
        predictions = None
        model_name = model_type.upper()

        # Try different models
        if model_type == "arima":
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(y_train, order=(2, 1, 2)).fit()
                predictions = model.forecast(steps=len(y_test))
                model_name = "ARIMA(2,1,2)"
            except Exception as e:
                print(f"ARIMA failed: {e}")
                model_type = "ar"

        if model_type == "ar" or predictions is None:
            try:
                from statsmodels.tsa.ar_model import AutoReg
                lags = min(15, len(y_train) // 4)
                model = AutoReg(y_train, lags=lags).fit()
                predictions = model.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)
                model_name = f"AR({lags})"
            except Exception as e:
                print(f"AR failed: {e}")
                model_type = "exp_smoothing"

        if model_type == "exp_smoothing" or predictions is None:
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                model = ExponentialSmoothing(y_train, trend='add').fit()
                predictions = model.forecast(len(y_test))
                model_name = "ExpSmoothing"
            except Exception as e:
                print(f"ExpSmoothing failed: {e}")
                # Last resort: naive forecast (repeat last value)
                predictions = np.full(len(y_test), y_train[-1])
                model_name = "Naive"

        predictions = np.array(predictions).flatten()

        # Compute metrics
        metrics = {
            'mse': float(mean_squared_error(y_test, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
            'mae': float(mean_absolute_error(y_test, predictions)),
            'r2': float(r2_score(y_test, predictions)),
        }

        # Plot: train + test + predictions with metrics
        if show_plot:
            plt.figure(figsize=(12, 6))
            plt.plot(x_train, y_train, 'b-', label='Train', linewidth=1)
            plt.plot(x_test, y_test, 'g-', label='Test (actual)', linewidth=1.5)
            plt.plot(x_test, predictions, 'r--', label=f'Predictions ({model_name})', linewidth=2)
            plt.axvline(x=split_idx, color='gray', linestyle='--', alpha=0.5, label='Train/Test split')
            plt.title(f'{description}\nRMSE={metrics["rmse"]:.4f}, MAE={metrics["mae"]:.4f}, R²={metrics["r2"]:.4f}')
            plt.xlabel('Time (index)')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return model, predictions, metrics


# Quick test
if __name__ == "__main__":
    # Sample data
    np.random.seed(42)
    t = np.linspace(0, 4 * np.pi, 200)
    y = 5 * np.sin(t) + 0.5 * t + np.random.normal(0, 0.5, 200)

    ts_data = {
        'x': t,
        'y': y,
        'description': 'Synthetic sine with trend'
    }

    # Sample report
    sample_report = {
        "summary": "Time series with upward trend and periodic component",
        "state_space_recommendation": {
            "method": "SSA",
            "embed_dim": 3,
            "delay": 1
        },
        "recommended_models": [
            {"name": "ARIMA", "reason": "Good for trending data"}
        ]
    }

    try:
        codegen = LLMCodeGenerator()
        print("Generating code...")

        code = codegen.generate_code(ts_data, sample_report)
        print("Generated code:")
        print("-" * 50)
        print(code[:500] + "...")
        print("-" * 50)

        print("\nExecuting code...")
        result = codegen.execute_code(ts_data)

        if result.success:
            print(f"Success! Metrics: {result.metrics}")
        else:
            print(f"Failed: {result.error_message}")
            print("\nUsing fallback processor...")
            model, preds, metrics = FallbackProcessor.process_timeseries(t, y)
            print(f"Fallback metrics: {metrics}")

    except Exception as e:
        print(f"Error: {e}")
