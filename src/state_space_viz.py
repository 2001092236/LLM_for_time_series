"""
State Space Visualization for Time Series
Implements various transformations and visualizations:
- Phase Space Embedding (Time-delay)
- SSA (Singular Spectrum Analysis)
- Fourier Transform
- Wavelet Transform
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Union
from dataclasses import dataclass

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class SSAResult:
    """Container for SSA decomposition results"""
    components: np.ndarray  # Reconstructed components
    singular_values: np.ndarray
    explained_variance: np.ndarray
    trend: np.ndarray
    seasonal: Optional[np.ndarray]
    residual: np.ndarray


class StateSpaceVisualizer:
    """
    Visualization of time series in state space representations
    """

    def __init__(self, y: np.ndarray, x: Optional[np.ndarray] = None):
        """
        Args:
            y: Time series values
            x: Time points (optional)
        """
        self.y = np.array(y).flatten()
        self.x = x if x is not None else np.arange(len(y))

    # ==================== PHASE SPACE EMBEDDING ====================

    def phase_space_embedding(self, embed_dim: int = 3, delay: int = 1) -> np.ndarray:
        """
        Create time-delay embedding of the time series

        Args:
            embed_dim: Embedding dimension (2 or 3 for visualization)
            delay: Time delay (lag) for embedding

        Returns:
            Embedded trajectory matrix (n_points x embed_dim)
        """
        n = len(self.y) - (embed_dim - 1) * delay
        embedded = np.zeros((n, embed_dim))

        for i in range(embed_dim):
            embedded[:, i] = self.y[i * delay:i * delay + n]

        return embedded

    def plot_phase_space(
        self,
        embed_dim: int = 3,
        delay: int = 1,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        save_path: Optional[str] = None,
        interactive: bool = True
    ) -> Union[plt.Figure, 'go.Figure']:
        """
        Plot phase space embedding

        Args:
            embed_dim: Dimension (2 or 3)
            delay: Time delay
            figsize: Figure size
            cmap: Colormap
            save_path: Path to save figure
            interactive: Use Plotly for interactive 3D (default True for dim=3)

        Returns:
            matplotlib Figure or plotly Figure
        """
        embedded = self.phase_space_embedding(embed_dim, delay)
        n = len(embedded)

        # Use interactive Plotly for 3D if available
        if embed_dim == 3 and interactive and PLOTLY_AVAILABLE:
            return self._plot_phase_space_3d_interactive(embedded, delay)

        # Fallback to matplotlib
        fig = plt.figure(figsize=figsize)

        if embed_dim == 2:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1],
                c=range(n), cmap=cmap, alpha=0.6, s=10
            )
            ax.plot(embedded[:, 0], embedded[:, 1], 'k-', alpha=0.1, linewidth=0.5)
            ax.set_xlabel(f'y(t)')
            ax.set_ylabel(f'y(t+{delay})')
            plt.colorbar(scatter, label='Time index')

        elif embed_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(
                embedded[:, 0], embedded[:, 1], embedded[:, 2],
                c=range(n), cmap=cmap, alpha=0.6, s=10
            )
            ax.plot(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                   'k-', alpha=0.1, linewidth=0.5)
            ax.set_xlabel(f'y(t)')
            ax.set_ylabel(f'y(t+{delay})')
            ax.set_zlabel(f'y(t+{2*delay})')
            plt.colorbar(scatter, label='Time index', shrink=0.6)

        ax.set_title(f'Phase Space Embedding (dim={embed_dim}, delay={delay})')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def _plot_phase_space_3d_interactive(
        self,
        embedded: np.ndarray,
        delay: int = 1
    ) -> 'go.Figure':
        """
        Create interactive 3D phase space plot with Plotly

        Args:
            embedded: Embedded trajectory (n x 3)
            delay: Time delay used

        Returns:
            Plotly Figure
        """
        n = len(embedded)
        time_indices = np.arange(n)

        # Create 3D scatter plot
        fig = go.Figure()

        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=embedded[:, 0],
            y=embedded[:, 1],
            z=embedded[:, 2],
            mode='lines',
            line=dict(color='rgba(100,100,100,0.3)', width=1),
            name='Trajectory',
            hoverinfo='skip'
        ))

        # Add scatter points with color by time
        fig.add_trace(go.Scatter3d(
            x=embedded[:, 0],
            y=embedded[:, 1],
            z=embedded[:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=time_indices,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(
                    title='Time',
                    thickness=15,
                    len=0.7
                )
            ),
            text=[f'Time: {i}<br>y(t): {embedded[i,0]:.3f}<br>y(t+{delay}): {embedded[i,1]:.3f}<br>y(t+{2*delay}): {embedded[i,2]:.3f}'
                  for i in range(n)],
            hoverinfo='text',
            name='Points'
        ))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Phase Space Embedding (3D, delay={delay})',
                x=0.5,
                font=dict(size=16)
            ),
            scene=dict(
                xaxis_title=f'y(t)',
                yaxis_title=f'y(t+{delay})',
                zaxis_title=f'y(t+{2*delay})',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=800,
            height=700,
            showlegend=False,
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    # ==================== SSA (Singular Spectrum Analysis) ====================

    def ssa_decompose(self, window_length: Optional[int] = None, n_components: int = 10) -> SSAResult:
        """
        Perform SSA decomposition

        Args:
            window_length: Window length L (default: N/2)
            n_components: Number of components to keep

        Returns:
            SSAResult with decomposed components
        """
        n = len(self.y)

        if window_length is None:
            window_length = n // 2

        window_length = min(window_length, n // 2)
        k = n - window_length + 1

        # Step 1: Embedding (trajectory matrix)
        trajectory = np.zeros((window_length, k))
        for i in range(window_length):
            trajectory[i, :] = self.y[i:i + k]

        # Step 2: SVD
        U, S, Vt = np.linalg.svd(trajectory, full_matrices=False)

        # Keep only n_components
        n_components = min(n_components, len(S))

        # Step 3: Grouping and reconstruction
        components = []
        for i in range(n_components):
            # Reconstruct each elementary matrix
            elem_matrix = S[i] * np.outer(U[:, i], Vt[i, :])

            # Diagonal averaging (Hankelization)
            reconstructed = self._hankel_to_ts(elem_matrix, n)
            components.append(reconstructed)

        components = np.array(components)

        # Explained variance
        total_var = np.sum(S ** 2)
        explained_variance = (S[:n_components] ** 2) / total_var

        # Trend: first component(s)
        trend = components[0] if len(components) > 0 else np.zeros(n)

        # Seasonal: next few components
        if len(components) > 2:
            seasonal = np.sum(components[1:3], axis=0)
        else:
            seasonal = None

        # Residual
        reconstructed_sum = np.sum(components, axis=0)
        residual = self.y - reconstructed_sum

        return SSAResult(
            components=components,
            singular_values=S[:n_components],
            explained_variance=explained_variance,
            trend=trend,
            seasonal=seasonal,
            residual=residual
        )

    def _hankel_to_ts(self, matrix: np.ndarray, n: int) -> np.ndarray:
        """Convert Hankel matrix back to time series via diagonal averaging"""
        L, K = matrix.shape
        ts = np.zeros(n)
        counts = np.zeros(n)

        for i in range(L):
            for j in range(K):
                idx = i + j
                if idx < n:
                    ts[idx] += matrix[i, j]
                    counts[idx] += 1

        return ts / counts

    def plot_ssa(
        self,
        window_length: Optional[int] = None,
        n_components: int = 5,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot SSA decomposition results
        """
        ssa = self.ssa_decompose(window_length, n_components)

        fig, axes = plt.subplots(3, 2, figsize=figsize)

        # Original series
        axes[0, 0].plot(self.x, self.y, 'b-', linewidth=0.8)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].grid(True, alpha=0.3)

        # Singular values
        axes[0, 1].semilogy(ssa.singular_values, 'ko-', markersize=5)
        axes[0, 1].set_title('Singular Values (log scale)')
        axes[0, 1].set_xlabel('Component')
        axes[0, 1].grid(True, alpha=0.3)

        # Trend component
        axes[1, 0].plot(self.x, ssa.trend, 'r-', linewidth=1.5)
        axes[1, 0].set_title(f'Trend (Component 1, var={ssa.explained_variance[0]:.1%})')
        axes[1, 0].grid(True, alpha=0.3)

        # Seasonal component
        if ssa.seasonal is not None:
            axes[1, 1].plot(self.x, ssa.seasonal, 'g-', linewidth=1)
            axes[1, 1].set_title('Seasonal (Components 2-3)')
        else:
            axes[1, 1].text(0.5, 0.5, 'No seasonal component', ha='center', va='center')
        axes[1, 1].grid(True, alpha=0.3)

        # Residual
        axes[2, 0].plot(self.x, ssa.residual, 'gray', linewidth=0.5)
        axes[2, 0].set_title('Residual')
        axes[2, 0].grid(True, alpha=0.3)

        # Explained variance
        cumvar = np.cumsum(ssa.explained_variance)
        axes[2, 1].bar(range(len(ssa.explained_variance)), ssa.explained_variance, alpha=0.7, label='Individual')
        axes[2, 1].plot(range(len(cumvar)), cumvar, 'ro-', markersize=5, label='Cumulative')
        axes[2, 1].set_title('Explained Variance by Component')
        axes[2, 1].set_xlabel('Component')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    # ==================== FOURIER TRANSFORM ====================

    def fourier_analysis(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform FFT analysis

        Returns:
            frequencies, magnitudes, phases
        """
        n = len(self.y)
        y_centered = self.y - np.mean(self.y)

        # FFT
        fft_result = np.fft.fft(y_centered)
        frequencies = np.fft.fftfreq(n)

        # Only positive frequencies
        positive_mask = frequencies >= 0
        frequencies = frequencies[positive_mask]
        fft_result = fft_result[positive_mask]

        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result)

        return frequencies, magnitudes, phases

    def plot_fourier(
        self,
        figsize: Tuple[int, int] = (12, 8),
        top_n_peaks: int = 5,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot Fourier analysis results
        """
        frequencies, magnitudes, phases = self.fourier_analysis()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Original signal
        axes[0, 0].plot(self.x, self.y, 'b-', linewidth=0.8)
        axes[0, 0].set_title('Original Time Series')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].grid(True, alpha=0.3)

        # Magnitude spectrum
        axes[0, 1].plot(frequencies[1:], magnitudes[1:], 'b-', linewidth=0.8)
        axes[0, 1].set_title('Magnitude Spectrum')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)

        # Find top peaks
        peak_indices = np.argsort(magnitudes[1:])[-top_n_peaks:] + 1
        for idx in peak_indices:
            axes[0, 1].axvline(x=frequencies[idx], color='r', linestyle='--', alpha=0.5)
            axes[0, 1].annotate(
                f'{frequencies[idx]:.3f}',
                xy=(frequencies[idx], magnitudes[idx]),
                fontsize=8
            )

        # Phase spectrum
        axes[1, 0].plot(frequencies[1:], phases[1:], 'g-', linewidth=0.5)
        axes[1, 0].set_title('Phase Spectrum')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].set_ylabel('Phase (radians)')
        axes[1, 0].grid(True, alpha=0.3)

        # Power spectrum (log scale)
        power = magnitudes[1:] ** 2
        axes[1, 1].semilogy(frequencies[1:], power, 'purple', linewidth=0.8)
        axes[1, 1].set_title('Power Spectrum (log scale)')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].set_ylabel('Power')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    # ==================== COMPREHENSIVE VISUALIZATION ====================

    def plot_all(
        self,
        figsize: Tuple[int, int] = (16, 10),
        save_path: Optional[str] = None,
        interactive_3d: bool = True
    ) -> Tuple[plt.Figure, Optional['go.Figure']]:
        """
        Create comprehensive state space visualization

        Args:
            figsize: Figure size for matplotlib plots
            save_path: Path to save matplotlib figure
            interactive_3d: Return interactive Plotly 3D figure separately

        Returns:
            Tuple of (matplotlib_figure, plotly_3d_figure or None)
        """
        fig = plt.figure(figsize=figsize)

        # Use actual x coordinates
        x = self.x

        # 1. Original time series
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.plot(x, self.y, 'b-', linewidth=0.8)
        ax1.set_title('Original Time Series')
        ax1.set_xlabel('Time')
        ax1.grid(True, alpha=0.3)

        # 2. Phase space 2D
        ax2 = fig.add_subplot(2, 3, 2)
        embedded_2d = self.phase_space_embedding(2, 1)
        ax2.scatter(embedded_2d[:, 0], embedded_2d[:, 1],
                   c=range(len(embedded_2d)), cmap='viridis', alpha=0.5, s=5)
        ax2.set_title('Phase Space (2D)')
        ax2.set_xlabel('y(t)')
        ax2.set_ylabel('y(t+1)')

        # 3. Phase space 3D (static matplotlib version)
        ax3 = fig.add_subplot(2, 3, 3, projection='3d')
        embedded_3d = self.phase_space_embedding(3, 1)
        ax3.scatter(embedded_3d[:, 0], embedded_3d[:, 1], embedded_3d[:, 2],
                   c=range(len(embedded_3d)), cmap='viridis', alpha=0.5, s=5)
        ax3.set_title('Phase Space (3D) - see interactive')

        # 4. SSA decomposition (trend)
        ax4 = fig.add_subplot(2, 3, 4)
        ssa = self.ssa_decompose(n_components=3)
        ax4.plot(x[:len(ssa.trend)], ssa.trend, 'r-', linewidth=1.5, label='Trend (SSA)')
        ax4.plot(x, self.y, 'b-', alpha=0.3, linewidth=0.5, label='Original')
        ax4.set_title('SSA Trend Extraction')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5. Fourier spectrum
        ax5 = fig.add_subplot(2, 3, 5)
        frequencies, magnitudes, _ = self.fourier_analysis()
        ax5.plot(frequencies[1:len(frequencies)//4], magnitudes[1:len(magnitudes)//4], 'purple', linewidth=0.8)
        ax5.set_title('Fourier Spectrum')
        ax5.set_xlabel('Frequency')
        ax5.grid(True, alpha=0.3)

        # 6. SSA singular values
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.semilogy(ssa.singular_values, 'ko-', markersize=5)
        ax6.set_title('SSA Singular Values')
        ax6.set_xlabel('Component')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        # Create interactive 3D Plotly figure
        plotly_fig = None
        if interactive_3d and PLOTLY_AVAILABLE:
            plotly_fig = self._plot_phase_space_3d_interactive(embedded_3d, delay=1)

        return fig, plotly_fig


# Quick test
if __name__ == "__main__":
    # Generate test data: damped oscillator
    np.random.seed(42)
    t = np.linspace(0, 10, 1000)
    y = np.exp(-0.2 * t) * np.sin(2 * np.pi * t) + 0.1 * np.random.randn(len(t))

    viz = StateSpaceVisualizer(y, t)

    # Phase space
    fig1 = viz.plot_phase_space(embed_dim=3, delay=5)
    plt.savefig('outputs/visualizations/phase_space.png', dpi=150)

    # SSA
    fig2 = viz.plot_ssa()
    plt.savefig('outputs/visualizations/ssa_decomposition.png', dpi=150)

    # Fourier
    fig3 = viz.plot_fourier()
    plt.savefig('outputs/visualizations/fourier_analysis.png', dpi=150)

    # All in one
    fig4 = viz.plot_all()
    plt.savefig('outputs/visualizations/state_space_all.png', dpi=150)

    plt.show()
