"""Op-amp characterization using synthetic voltages exclusively."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
from pathlib import Path

# ==============================
# Global font settings - Times New Roman for all text including math
# ==============================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

warnings.filterwarnings('ignore')


class InvertingAmplifierAnalyzer:
    def __init__(self, theoretical_gain=-10):
        """
        Initialize Inverting Amplifier Analyzer

        Parameters:
        theoretical_gain: Theoretical gain value (-Rf/Rin = -100kΩ/10kΩ = -10)
        """
        self.theoretical_gain = theoretical_gain
        self.measured_gain = None
        self.saturation_threshold = 11.5  # Approximate reference only; not a measured limit

    def load_synthetic_data(self):
        """
        Load synthetic data
        """
        # Input voltage peak (V)
        v_i = np.array([0.22, 0.71, 1.48])

        v_o_simulated = np.clip(-9.8 * v_i - 0.03, -11.5, 11.5)

        return v_i.reshape(-1, 1), v_o_simulated

    def perform_linear_regression(self, v_i, v_o_simulated):
        """
        Perform linear regression analysis
        """
        # Method 1: Ordinary Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(v_i, v_o_simulated)
        gain_lr = lr_model.coef_[0]
        v_o_pred_lr = lr_model.predict(v_i)
        r2_lr = r2_score(v_o_simulated, v_o_pred_lr)

        # Method 2: Robust Regression (RANSAC) - more resistant to outliers
        ransac_model = RANSACRegressor(random_state=42)
        ransac_model.fit(v_i, v_o_simulated)
        gain_ransac = ransac_model.estimator_.coef_[0]
        v_o_pred_ransac = ransac_model.predict(v_i)

        # Handle potential None values in RANSAC predictions
        inlier_mask = ransac_model.inlier_mask_
        r2_ransac = r2_score(v_o_simulated[inlier_mask], v_o_pred_ransac[inlier_mask])

        return {
            'linear': {'gain': gain_lr, 'predictions': v_o_pred_lr, 'r2': r2_lr},
            'ransac': {'gain': gain_ransac, 'predictions': v_o_pred_ransac, 'r2': r2_ransac, 'inliers': inlier_mask}
        }

    def detect_anomalies(self, v_i, v_o_simulated, predictions):
        """
        Detect anomalies: gain loss, output clipping, etc.
        """
        anomalies = []

        # Detect saturation
        saturation_mask = np.abs(v_o_simulated) >= self.saturation_threshold * 0.95
        if np.any(saturation_mask):
            saturation_points = v_i.flatten()[saturation_mask]
            anomalies.append({
                'type': 'output_clipping',
                'points': saturation_points,
                'message': f'Possible output saturation near the assumed threshold at input voltages: {saturation_points} V; check the waveforms.'
            })

        # Detect gain loss
        actual_gains = v_o_simulated / v_i.flatten()
        gain_deviation = np.abs(actual_gains - self.theoretical_gain) / np.abs(self.theoretical_gain)
        gain_loss_mask = gain_deviation > 0.15  # More than 15% gain deviation

        if np.any(gain_loss_mask):
            gain_loss_points = v_i.flatten()[gain_loss_mask]
            anomalies.append({
                'type': 'gain_loss',
                'points': gain_loss_points,
                'message': f'Gain loss detected at input voltages: {gain_loss_points} V'
            })

        # Detect non-linearity
        residuals = v_o_simulated - predictions
        residual_std = np.std(residuals)
        if residual_std > 0.3:  # Residual standard deviation threshold
            anomalies.append({
                'type': 'non_linearity',
                'severity': residual_std,
                'message': f'Regression residual standard deviation: {residual_std:.3f} V (only 3 data points).'
            })

        return anomalies

    def analyze_circuit_performance(self):
        """
        Perform complete circuit performance analysis
        """
        print("=" * 60)
        print("INVERTING AMPLIFIER SYSTEM IDENTIFICATION ANALYSIS")
        print("=" * 60)

        # Load data
        v_i, v_o_simulated = self.load_synthetic_data()

        print(f"Input Voltage (V): {v_i.flatten()}")
        print(f"Simulated Output Voltage (V): {v_o_simulated}")
        print(f"Theoretical Gain: {self.theoretical_gain}")

        # Linear regression analysis
        regression_results = self.perform_linear_regression(v_i, v_o_simulated)

        # Use RANSAC results as primary analysis
        main_result = regression_results['ransac']
        self.measured_gain = main_result['gain']

        print(f"\n--- REGRESSION ANALYSIS RESULTS ---")
        print("Note: both regressions use all 3 points, including the likely saturated point.")
        print("With only 3 measurements, RANSAC inliers and R² are not reliable validation.")
        # Identify the linear-region points using the ideal ±12 V supply limit.
        linear_mask = np.abs(v_i.flatten() * self.theoretical_gain) < 12.0
        if np.count_nonzero(linear_mask) >= 2:
            linear_region_model = LinearRegression().fit(v_i[linear_mask], v_o_simulated[linear_mask])
            print(f"Linear-region gain (2 points, with intercept): {linear_region_model.coef_[0]:.3f}")
            print("Two points determine a line but do not independently validate the fit.")
        print(f"All-point gain (RANSAC): {self.measured_gain:.3f}")
        print(f"Gain Error: {abs((self.measured_gain - self.theoretical_gain) / self.theoretical_gain * 100):.2f}%")
        print(f"R² on RANSAC inliers: {main_result['r2']:.4f}")

        # Anomaly detection
        anomalies = self.detect_anomalies(v_i, v_o_simulated, main_result['predictions'])

        if anomalies:
            print(f"\n--- ANOMALY DETECTION REPORT ---")
            for anomaly in anomalies:
                print(f"⚠️  {anomaly['message']}")
        else:
            print(f"\n✅ Circuit operating normally, no significant anomalies detected")

        # The global fit includes a saturated point; assess the linear region separately.
        print("\n--- PERFORMANCE EVALUATION ---")
        print("Compare the linear-region gain with the theoretical gain; inspect waveforms for clipping.")

        return v_i, v_o_simulated, regression_results, anomalies

    def plot_results(self, v_i, v_o_simulated, regression_results):
        """
        Plot analysis results
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Input-Output characteristics plot
        v_i_continuous = np.linspace(0, 2, 100).reshape(-1, 1)
        v_o_ransac = regression_results['ransac']['predictions']
        v_o_linear = regression_results['linear']['predictions']

        # Plot synthetic data
        ax1.scatter(v_i, v_o_simulated, color='blue', s=80, label='Synthetic data', zorder=5)

        # Highlight RANSAC inliers and outliers
        inlier_mask = regression_results['ransac']['inliers']
        ax1.scatter(v_i[inlier_mask], v_o_simulated[inlier_mask],
                    color='green', s=100, alpha=0.6, label='RANSAC inliers')
        ax1.scatter(v_i[~inlier_mask], v_o_simulated[~inlier_mask],
                    color='red', s=100, alpha=0.6, label='RANSAC outliers')

        # Plot regression lines
        ax1.plot(v_i, v_o_ransac, 'r-', linewidth=2,
                 label=f'Robust regression (Gain: {regression_results["ransac"]["gain"]:.2f})')
        ax1.plot(v_i, v_o_linear, 'g--', linewidth=1,
                 label=f'Linear regression (Gain: {regression_results["linear"]["gain"]:.2f})')

        # Theoretical curve
        v_o_theoretical = v_i * self.theoretical_gain
        ax1.plot(v_i, v_o_theoretical, 'k:', linewidth=1,
                 label=f'Theoretical (gain: {self.theoretical_gain})')

        # Saturation region markers
        ax1.axhline(y=self.saturation_threshold, color='orange', linestyle='--',
                    alpha=0.7, label='Estimated positive reference (11.5 V)')
        ax1.axhline(y=-self.saturation_threshold, color='orange', linestyle='--',
                    alpha=0.7, label='Estimated negative reference (-11.5 V)')

        ax1.set_xlabel('Input voltage $v_i$ (V)', fontsize=16)
        ax1.set_ylabel('Output voltage $v_o$ (V)', fontsize=16)
        ax1.set_title('Inverting amplifier input-output characteristics', fontsize=18)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # Residual plot
        residuals = v_o_simulated - v_o_ransac
        ax2.scatter(v_i, residuals, color='red', s=80)
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.set_xlabel('Input voltage $v_i$ (V)', fontsize=16)
        ax2.set_ylabel('Residuals (V)', fontsize=16)
        ax2.set_title('Regression residual analysis', fontsize=18)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=14)

        # Add residual statistics
        ax2.text(0.95, 0.95, f'Residual Std: {np.std(residuals):.3f} V',
                 transform=ax2.transAxes, horizontalalignment='right', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        output_path = Path(__file__).resolve().with_name('Lab3-1_results.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
        plt.show()


def main():
    """
    Main function - Execute complete analysis pipeline
    """
    # Create analyzer instance (theoretical gain = -10)
    analyzer = InvertingAmplifierAnalyzer(theoretical_gain=-10)

    # Perform analysis
    v_i, v_o_simulated, regression_results, anomalies = analyzer.analyze_circuit_performance()

    # Plot results
    analyzer.plot_results(v_i, v_o_simulated, regression_results)

    # Generate analysis report
    print("\n" + "=" * 60)
    print("SYSTEM IDENTIFICATION ANALYSIS REPORT")
    print("=" * 60)
    print(f"Theoretical Gain: {analyzer.theoretical_gain}")
    print(f"All-point RANSAC Gain: {analyzer.measured_gain:.3f}")
    print(
        f"All-point gain error: {abs((analyzer.measured_gain - analyzer.theoretical_gain) / analyzer.theoretical_gain * 100):.2f}%")

    print("The all-point fit includes the likely saturated measurement; it is not the linear-region gain.")
    print("Theoretical values below are linear extrapolations and can exceed the supply range.")

    # Print detailed comparison table
    print(f"\n--- DETAILED COMPARISON TABLE ---")
    print(f"{'Input Vi (V)':<12} {'Simulated Vo (V)':<15} {'Theoretical Vo (V)':<18} {'Error (%)':<10}")
    print("-" * 55)
    for i, (vi, vo_meas) in enumerate(zip(v_i.flatten(), v_o_simulated)):
        vo_theo = vi * analyzer.theoretical_gain
        error_pct = abs((vo_meas - vo_theo) / vo_theo * 100) if vo_theo != 0 else 0
        print(f"{vi:<12.1f} {vo_meas:<15.3f} {vo_theo:<18.3f} {error_pct:<10.2f}")


if __name__ == "__main__":
    print("Dataset: synthetic voltages generated by a simple gain model.")
    main()