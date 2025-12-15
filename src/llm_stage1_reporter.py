"""
LLM Stage 1: Analytical Report Generator
Analyzes time series features and baseline model results,
generates recommendations for model selection and transformations.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMAnalyticalReporter:
    """
    Stage 1 of the LLM Agent: Generates analytical report
    based on extracted features and baseline model metrics.
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

    def _build_system_prompt(self) -> str:
        """Build system prompt for analytical report generation"""
        return """You are an expert time series analyst and data scientist.

Your task is to analyze statistical features and baseline model results for a time series,
then provide a structured analytical report with recommendations.

You must output your analysis in the following JSON structure:
{
    "summary": "Brief 2-3 sentence summary of the time series characteristics",
    "characteristics": {
        "trend": "Description of trend (none/weak/strong, direction)",
        "seasonality": "Description of seasonality (none/weak/strong, period)",
        "stationarity": "Is it stationary? Implications",
        "noise_level": "Low/Medium/High",
        "complexity": "Simple/Moderate/Complex"
    },
    "recommended_transformations": [
        {
            "name": "Transformation name (SSA, FFT, Differencing, etc.)",
            "reason": "Why this transformation is recommended",
            "priority": 1
        }
    ],
    "recommended_models": [
        {
            "name": "Model name (ARIMA, Prophet, LSTM, GPR, etc.)",
            "reason": "Why this model is suitable",
            "suggested_params": "Suggested parameters if any",
            "priority": 1
        }
    ],
    "state_space_recommendation": {
        "method": "SSA/Fourier/PhaseSpace/Kalman",
        "embed_dim": 3,
        "delay": 1,
        "reason": "Why this state space representation"
    },
    "warnings": ["Any potential issues or caveats"],
    "confidence": "High/Medium/Low"
}

Available State Space transformations:
- SSA (Singular Spectrum Analysis): Best for trend extraction and noise reduction
- Fourier Transform: Best for frequency analysis of periodic signals
- Phase Space Embedding: Best for visualizing dynamics and detecting chaos
- Kalman Filter: Best for noisy data with known state-space model

Available models to recommend:
- ARIMA: For non-stationary data with trend, requires differencing
- Prophet: For data with multiple seasonalities, handles holidays
- AutoReg (AR): For stationary data with autocorrelation
- ExponentialSmoothing: For trend and seasonality
- GPR (Gaussian Process): For uncertainty estimation, smooth functions
- LSTM: For complex nonlinear patterns (requires more data)

Always provide practical, actionable recommendations based on the data characteristics."""

    def _build_user_prompt(
        self,
        features: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        context: str = "",
        user_prompt: str = ""
    ) -> str:
        """Build user prompt with features and metrics"""

        features_json = json.dumps(features, indent=2, ensure_ascii=False, default=str)
        metrics_json = json.dumps(baseline_metrics, indent=2, ensure_ascii=False, default=str)

        prompt = f"""Analyze this time series and provide recommendations:

## EXTRACTED FEATURES
{features_json}

## BASELINE MODEL RESULTS
{metrics_json}

## CONTEXT
{context if context else "No additional context provided."}

## USER REQUEST
{user_prompt if user_prompt else "Provide general analysis and model recommendations."}

Please analyze the time series and provide your structured JSON report with:
1. Summary of characteristics
2. Recommended state space transformations
3. Recommended models with parameters
4. Any warnings or caveats

Respond ONLY with valid JSON matching the specified structure."""

        return prompt

    def generate_report(
        self,
        features: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        context: str = "",
        user_prompt: str = "",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Generate analytical report using LLM

        Args:
            features: Dictionary of extracted features from TSFeatureExtractor
            baseline_metrics: Dictionary of baseline model metrics
            context: Physical/domain context of the data
            user_prompt: Specific user request (e.g., "Find change points")
            temperature: LLM temperature (lower = more deterministic)

        Returns:
            Dictionary with structured analytical report
        """
        system_prompt = self._build_system_prompt()
        user_prompt_full = self._build_user_prompt(features, baseline_metrics, context, user_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt_full}
                ],
                temperature=temperature,
                max_tokens=10000,
                response_format={"type": "json_object"}
            )

            # Parse JSON response
            content = response.choices[0].message.content
            report = json.loads(content)

            # Add metadata
            report["_metadata"] = {
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else None,
            }

            return report

        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse LLM response as JSON: {e}",
                "raw_response": content if 'content' in locals() else None
            }
        except Exception as e:
            return {
                "error": f"LLM API error: {e}"
            }

    def generate_text_report(
        self,
        features: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        context: str = "",
        user_prompt: str = ""
    ) -> str:
        """
        Generate human-readable text report

        Returns:
            Formatted text report
        """
        report = self.generate_report(features, baseline_metrics, context, user_prompt)

        if "error" in report:
            return f"Error generating report: {report['error']}"

        # Format as text
        text = []
        text.append("=" * 60)
        text.append("TIME SERIES ANALYTICAL REPORT")
        text.append("=" * 60)
        text.append("")

        # Summary
        text.append("## SUMMARY")
        text.append(report.get("summary", "N/A"))
        text.append("")

        # Characteristics
        text.append("## CHARACTERISTICS")
        chars = report.get("characteristics", {})
        for key, value in chars.items():
            text.append(f"  - {key.replace('_', ' ').title()}: {value}")
        text.append("")

        # State Space Recommendation
        text.append("## STATE SPACE RECOMMENDATION")
        ss = report.get("state_space_recommendation", {})
        text.append(f"  Method: {ss.get('method', 'N/A')}")
        text.append(f"  Embedding dim: {ss.get('embed_dim', 'N/A')}")
        text.append(f"  Delay: {ss.get('delay', 'N/A')}")
        text.append(f"  Reason: {ss.get('reason', 'N/A')}")
        text.append("")

        # Recommended Transformations
        text.append("## RECOMMENDED TRANSFORMATIONS")
        transforms = report.get("recommended_transformations", [])
        for i, t in enumerate(transforms, 1):
            text.append(f"  {i}. {t.get('name', 'N/A')}")
            text.append(f"     Reason: {t.get('reason', 'N/A')}")
        text.append("")

        # Recommended Models
        text.append("## RECOMMENDED MODELS")
        models = report.get("recommended_models", [])
        for i, m in enumerate(models, 1):
            text.append(f"  {i}. {m.get('name', 'N/A')}")
            text.append(f"     Reason: {m.get('reason', 'N/A')}")
            if m.get('suggested_params'):
                text.append(f"     Params: {m.get('suggested_params')}")
        text.append("")

        # Warnings
        warnings = report.get("warnings", [])
        if warnings:
            text.append("## WARNINGS")
            for w in warnings:
                text.append(f"  - {w}")
            text.append("")

        # Confidence
        text.append(f"## CONFIDENCE: {report.get('confidence', 'N/A')}")
        text.append("=" * 60)

        return "\n".join(text)

    def save_report(self, report: Dict[str, Any], filepath: str):
        """Save report to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)


# Quick test
if __name__ == "__main__":
    # Test with sample data
    sample_features = {
        "length": 500,
        "mean": 2.5,
        "std": 1.2,
        "trend_strength": 0.65,
        "trend_direction": "upward",
        "seasonal_strength": 0.3,
        "estimated_period": 50,
        "is_stationary": False,
        "acf_lag1": 0.85,
        "has_strong_autocorr": True,
    }

    sample_metrics = {
        "arima": {"rmse": 0.45, "mae": 0.35, "aic": 250.5},
        "autoreg": {"rmse": 0.52, "mae": 0.40},
        "exp_smoothing": {"rmse": 0.48, "mae": 0.38},
    }

    try:
        reporter = LLMAnalyticalReporter()
        print("Generating report...")

        text_report = reporter.generate_text_report(
            sample_features,
            sample_metrics,
            context="Synthetic time series with trend and seasonality",
            user_prompt="Recommend the best model for forecasting"
        )

        print(text_report)

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure OPENAI_API_KEY is set in .env file")
