"""
Main Pipeline: LLM Time Series Agent
Orchestrates the full two-stage pipeline:
  Stage 1: Data → Features → Baseline → LLM Analytical Report
  Stage 2: Report → LLM Code Generation → Execution → Results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import project modules
from data_generator import DataGenerator, TimeSeriesData
from feature_extractor import TSFeatureExtractor
from baseline_models import BaselineModels
from state_space_viz import StateSpaceVisualizer
from llm_stage1_reporter import LLMAnalyticalReporter
from llm_stage2_codegen import LLMCodeGenerator, FallbackProcessor


@dataclass
class PipelineResult:
    """Container for complete pipeline results"""
    # Input
    dataset_name: str
    data_description: str
    data_context: str

    # Features
    features: Dict[str, Any]

    # Baseline models
    baseline_metrics: Dict[str, Any]
    best_baseline_model: str
    best_baseline_rmse: float

    # LLM Stage 1
    llm_report: Dict[str, Any]

    # LLM Stage 2
    generated_code: str
    final_model: Any
    final_predictions: np.ndarray
    final_metrics: Dict[str, float]

    # Status
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding non-serializable objects)"""
        d = asdict(self)
        d['final_model'] = str(type(self.final_model).__name__) if self.final_model else None
        d['final_predictions'] = self.final_predictions.tolist() if len(self.final_predictions) > 0 else []
        return d

    def save(self, filepath: str):
        """Save results to JSON"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)


class TimeSeriesAgent:
    """
    Two-stage LLM Agent for Time Series Analysis

    Pipeline:
    1. Load data
    2. Extract features
    3. Run baseline models
    4. LLM Stage 1: Generate analytical report
    5. LLM Stage 2: Generate and execute processing code
    6. Visualize results
    """

    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.data_generator = DataGenerator()

        # Initialize LLM components (lazy loading)
        self._reporter = None
        self._codegen = None

    @property
    def reporter(self) -> LLMAnalyticalReporter:
        if self._reporter is None:
            self._reporter = LLMAnalyticalReporter()
        return self._reporter

    @property
    def codegen(self) -> LLMCodeGenerator:
        if self._codegen is None:
            self._codegen = LLMCodeGenerator()
        return self._codegen

    def log(self, message: str, level: str = "INFO"):
        """Print log message if verbose"""
        if self.verbose:
            icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERROR": "❌", "STEP": "📌"}
            icon = icons.get(level, "")
            print(f"{icon} [{level}] {message}")

    def run(
        self,
        dataset_name: str = "synthetic_sine",
        context: str = "",
        user_prompt: str = "",
        save_results: bool = True,
        show_plots: bool = True,
        interactive: bool = True,
        **dataset_kwargs
    ) -> PipelineResult:
        """
        Run the complete pipeline

        Args:
            dataset_name: Name of dataset to load (see DataGenerator.get_all_datasets())
            context: Physical/domain context for the data
            user_prompt: Specific user request (e.g., "Find change points")
            save_results: Save results and plots to outputs/
            show_plots: Display plots
            interactive: Use interactive Plotly 3D visualizations (default True)
            **dataset_kwargs: Additional arguments for dataset generator

        Returns:
            PipelineResult with all outputs
        """
        import time
        start_time = time.time()

        self.log(f"{'='*60}", "INFO")
        self.log(f"LLM TIME SERIES AGENT - Processing: {dataset_name}", "STEP")
        self.log(f"{'='*60}", "INFO")

        # ========== STEP 1: Load Data ==========
        self.log("Loading data...", "STEP")
        try:
            ts_data = self.data_generator.load(dataset_name, **dataset_kwargs)
            self.log(f"Loaded {len(ts_data.y)} points: {ts_data.description}", "OK")
        except Exception as e:
            return self._error_result(dataset_name, f"Data loading failed: {e}")

        # ========== STEP 2: Extract Features ==========
        self.log("Extracting features...", "STEP")
        try:
            extractor = TSFeatureExtractor(ts_data.y, ts_data.x)
            features = extractor.extract_all()
            self.log(f"Extracted {len(features)} features", "OK")

            if self.verbose:
                print(extractor.get_summary())
        except Exception as e:
            return self._error_result(dataset_name, f"Feature extraction failed: {e}")

        # ========== STEP 3: Run Baseline Models ==========
        self.log("Running baseline models...", "STEP")
        try:
            baseline = BaselineModels(ts_data.y, ts_data.x, train_ratio=0.8)
            baseline.run_all()
            baseline_metrics = baseline.get_metrics_summary()

            best = baseline.get_best_model()
            best_name = best.name if best else "None"
            best_rmse = best.metrics.get('rmse', float('inf')) if best else float('inf')

            if self.verbose:
                baseline.print_summary()

            self.log(f"Best baseline: {best_name} (RMSE={best_rmse:.4f})", "OK")
        except Exception as e:
            self.log(f"Baseline models failed: {e}", "WARN")
            baseline_metrics = {}
            best_name = "None"
            best_rmse = float('inf')

        # ========== STEP 4: LLM Stage 1 - Analytical Report ==========
        self.log("Generating LLM analytical report (Stage 1)...", "STEP")
        try:
            full_context = f"{ts_data.context}\n{context}" if context else ts_data.context

            llm_report = self.reporter.generate_report(
                features=features,
                baseline_metrics=baseline_metrics,
                context=full_context,
                user_prompt=user_prompt
            )

            if "error" in llm_report:
                raise Exception(llm_report["error"])

            self.log("Analytical report generated", "OK")

            if self.verbose:
                text_report = self.reporter.generate_text_report(
                    features, baseline_metrics, full_context, user_prompt
                )
                print(text_report)

        except Exception as e:
            self.log(f"LLM Stage 1 failed: {e}", "WARN")
            # Create minimal fallback report
            llm_report = {
                "summary": "Analysis failed, using defaults",
                "state_space_recommendation": {"method": "SSA", "embed_dim": 3, "delay": 1},
                "recommended_models": [{"name": "ARIMA", "reason": "Default fallback"}],
            }

        # ========== STEP 5: LLM Stage 2 - Code Generation ==========
        self.log("Generating processing code (Stage 2)...", "STEP")
        ts_dict = ts_data.to_dict()

        try:
            code = self.codegen.generate_code(ts_dict, llm_report)
            self.log(f"Generated {len(code)} chars of code", "OK")

            if self.verbose:
                print("\n--- Generated Code Preview ---")
                print(code[:800] + "..." if len(code) > 800 else code)
                print("--- End Preview ---\n")

        except Exception as e:
            self.log(f"Code generation failed: {e}", "WARN")
            code = ""

        # ========== STEP 6: Execute Generated Code ==========
        self.log("Executing generated code...", "STEP")
        try:
            result = self.codegen.execute_code(ts_dict)

            if result.success:
                final_model = result.model
                final_predictions = result.predictions
                final_metrics = result.metrics
                self.log(f"Code executed successfully! Metrics: {final_metrics}", "OK")
            else:
                raise Exception(result.error_message)

        except Exception as e:
            self.log(f"Execution failed: {e}. Using fallback processor...", "WARN")

            # Fallback processing
            model_type = llm_report.get('recommended_models', [{}])[0].get('name', 'arima').lower()
            final_model, final_predictions, final_metrics = FallbackProcessor.process_timeseries(
                ts_data.x, ts_data.y, ts_data.description, model_type
            )
            self.log(f"Fallback completed. Metrics: {final_metrics}", "OK")

        # ========== STEP 7: Visualizations ==========
        self.log("Creating visualizations...", "STEP")

        # State Space visualization
        viz = StateSpaceVisualizer(ts_data.y, ts_data.x)

        ss_rec = llm_report.get('state_space_recommendation', {})
        embed_dim = ss_rec.get('embed_dim', 3)
        delay = ss_rec.get('delay', 1)

        # Get comprehensive state space visualizations
        # Returns: (matplotlib_figure, plotly_3d_figure or None)
        fig_ss_mpl, fig_ss_plotly = viz.plot_all(interactive_3d=interactive)

        # Forecast visualization (using actual x coordinates)
        fig_forecast = self._plot_forecast(
            ts_data.y, final_predictions,
            x=ts_data.x, train_ratio=0.8, title=f"{dataset_name} - Forecast"
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if save_results:
            base_path = f"outputs/visualizations/{dataset_name}_{timestamp}"

            # Save matplotlib state space figure
            fig_ss_mpl.savefig(f"{base_path}_state_space.png", dpi=150, bbox_inches='tight')

            # Save Plotly figure as HTML (interactive)
            if fig_ss_plotly is not None:
                try:
                    fig_ss_plotly.write_html(f"{base_path}_state_space_3d.html")
                except Exception:
                    pass  # Plotly not available or error

            fig_forecast.savefig(f"{base_path}_forecast.png", dpi=150, bbox_inches='tight')
            self.log(f"Plots saved to {base_path}_*.png/html", "OK")

        if show_plots:
            # Show matplotlib figures
            plt.show()

            # Show interactive Plotly 3D figure
            if interactive and fig_ss_plotly is not None:
                try:
                    fig_ss_plotly.show()
                except Exception:
                    pass
        else:
            plt.close('all')

        # ========== STEP 8: Save Results ==========
        execution_time = time.time() - start_time

        pipeline_result = PipelineResult(
            dataset_name=dataset_name,
            data_description=ts_data.description,
            data_context=ts_data.context,
            features=features,
            baseline_metrics=baseline_metrics,
            best_baseline_model=best_name,
            best_baseline_rmse=best_rmse,
            llm_report=llm_report,
            generated_code=code if code else "Fallback used",
            final_model=final_model,
            final_predictions=final_predictions,
            final_metrics=final_metrics,
            success=True,
            execution_time=execution_time
        )

        if save_results:
            report_path = f"outputs/reports/{dataset_name}_{timestamp}.json"
            pipeline_result.save(report_path)
            self.log(f"Results saved to {report_path}", "OK")

        self.log(f"Pipeline completed in {execution_time:.2f}s", "OK")
        self.log(f"{'='*60}", "INFO")

        return pipeline_result

    def _plot_forecast(
        self,
        y: np.ndarray,
        predictions: np.ndarray,
        x: Optional[np.ndarray] = None,
        train_ratio: float = 0.8,
        title: str = "Forecast"
    ) -> plt.Figure:
        """Create forecast visualization using actual x coordinates"""
        n = len(y)

        # Use provided x coordinates or integer indices as fallback
        if x is None:
            x = np.arange(n)

        split_idx = int(n * train_ratio)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Original data
        ax.plot(x[:split_idx], y[:split_idx], 'b-', label='Train', linewidth=1)
        ax.plot(x[split_idx:], y[split_idx:], 'g-', label='Test (actual)', linewidth=1)

        # Predictions
        if len(predictions) > 0:
            pred_x = x[split_idx:split_idx + len(predictions)]
            ax.plot(pred_x, predictions, 'r--', label='Predictions', linewidth=2)

        # Train/test split line
        ax.axvline(x=x[split_idx], color='gray', linestyle='--', alpha=0.7, label='Split')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def _error_result(self, dataset_name: str, error_msg: str) -> PipelineResult:
        """Create error result"""
        self.log(error_msg, "ERROR")
        return PipelineResult(
            dataset_name=dataset_name,
            data_description="",
            data_context="",
            features={},
            baseline_metrics={},
            best_baseline_model="",
            best_baseline_rmse=float('inf'),
            llm_report={},
            generated_code="",
            final_model=None,
            final_predictions=np.array([]),
            final_metrics={},
            success=False,
            error_message=error_msg
        )


def run_demo():
    """Run demo on multiple datasets"""
    agent = TimeSeriesAgent(verbose=True)

    datasets = [
        ("synthetic_sine", "Superposition of sine waves"),
        ("physics_damped_oscillator", "Physical damped pendulum"),
        ("real_airline", "Historical airline passengers"),
    ]

    results = []

    for name, context in datasets:
        print(f"\n\n{'#'*70}")
        print(f"# PROCESSING: {name}")
        print(f"{'#'*70}\n")

        result = agent.run(
            dataset_name=name,
            context=context,
            user_prompt="Analyze and recommend the best forecasting approach",
            save_results=True,
            show_plots=False
        )

        results.append(result)

        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        if result.success:
            print(f"Final metrics: {result.final_metrics}")

    return results


if __name__ == "__main__":
    # Single dataset run
    agent = TimeSeriesAgent(verbose=True)

    result = agent.run(
        dataset_name="synthetic_sine",
        context="Synthetic time series for testing",
        user_prompt="Analyze the time series and recommend the best model",
        save_results=True,
        show_plots=True
    )

    if result.success:
        print("\n✅ Pipeline completed successfully!")
        print(f"Final metrics: {result.final_metrics}")
    else:
        print(f"\n❌ Pipeline failed: {result.error_message}")
