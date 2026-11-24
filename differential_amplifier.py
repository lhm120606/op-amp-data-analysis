"""Op-amp characterization using synthetic voltages exclusively."""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
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


class DifferentialAmplifierAnalyzer:
    def __init__(self, theoretical_gain=10.0):
        """
        Initialize Differential Amplifier Analyzer

        Parameters:
        theoretical_gain: Theoretical differential gain (Rf/Rin)
        """
        self.theoretical_gain = theoretical_gain
        self.model = None
        self.coefficients = None
        self.saturation_threshold = 11.5

    def load_synthetic_data(self):
        """
        Load synthetic data for differential amplifier
        """
        v_i1 = np.array([0.15, 0.55, 0.95, 0.35])  # vi1
        v_i2 = np.array([0.72, 0.41, 0.32, 0.84])  # vi2

        v_o_simulated = 9.7 * (v_i2 - v_i1) + 0.05 * (v_i1 + v_i2) + 0.02

        return v_i1, v_i2, v_o_simulated

    def perform_multiple_linear_regression(self, v_i1, v_i2, v_o_simulated):
        """
        Perform multiple linear regression with enhanced diagnostics
        """
        # Prepare feature matrix
        X = np.column_stack((v_i1, v_i2))  # Intercept is fitted by the model

        # Method 1: Ordinary Least Squares
        ols_model = LinearRegression()
        ols_model.fit(X, v_o_simulated)
        ols_coef = ols_model.coef_
        ols_intercept = ols_model.intercept_

        # Method 2: Ridge Regression for better stability
        ridge_model = Ridge(alpha=0.1)
        ridge_model.fit(X, v_o_simulated)
        ridge_coef = ridge_model.coef_
        ridge_intercept = ridge_model.intercept_

        # Method 3: Physically constrained regression (a ≈ -b)
        X_constrained = (v_i2 - v_i1).reshape(-1, 1)  # Intercept is fitted by the model
        constrained_model = LinearRegression()
        constrained_model.fit(X_constrained, v_o_simulated)
        constrained_gain = constrained_model.coef_[0]
        constrained_offset = constrained_model.intercept_

        # Four samples cannot support three-fold R²: some test folds have one point.
        ols_scores = None
        ridge_scores = None

        predictions = {
            'ols': ols_model.predict(X),
            'ridge': ridge_model.predict(X),
            'constrained': constrained_model.predict(X_constrained)
        }

        return {
            'ols': {'coefficients': np.append(ols_coef, ols_intercept),
                    'cv_scores': ols_scores,
                    'predictions': predictions['ols']},
            'ridge': {'coefficients': np.append(ridge_coef, ridge_intercept),
                      'cv_scores': ridge_scores,
                      'predictions': predictions['ridge']},
            'constrained': {'gain': constrained_gain, 'offset': constrained_offset,
                            'predictions': predictions['constrained']},
            'feature_matrix': X
        }

    def detect_anomalies(self, v_i1, v_i2, v_o_simulated, regression_results):
        """
        Comprehensive anomaly detection for differential amplifier
        """
        anomalies = []
        predictions = regression_results['ridge']['predictions']

        # 1. Detect saturation
        saturation_mask = np.abs(v_o_simulated) >= self.saturation_threshold * 0.9
        if np.any(saturation_mask):
            anomalies.append({
                'type': 'output_saturation',
                'count': np.sum(saturation_mask),
                'message': f'Possible saturation in {np.sum(saturation_mask)} measurements; inspect the waveforms.'
            })

        # Common-mode and differential inputs vary together in these four points.
        # Correlation alone cannot identify common-mode gain, so do not flag it.

        # 3. Detect coefficient symmetry deviation
        ols_coef = regression_results['ols']['coefficients']
        symmetry_deviation = abs(ols_coef[0] + ols_coef[1]) / (abs(ols_coef[0]) + abs(ols_coef[1]) + 1e-8)
        if symmetry_deviation > 0.2:
            anomalies.append({
                'type': 'coefficient_asymmetry',
                'deviation': symmetry_deviation,
                'message': f'Significant coefficient asymmetry: {symmetry_deviation:.3f} (ideal: a = -b)'
            })

        # 4. Detect offset voltage
        offset = regression_results['ols']['coefficients'][2]
        if abs(offset) > 0.1:
            anomalies.append({
                'type': 'dc_offset',
                'value': offset,
                'message': f'Significant DC offset detected: {offset:.3f} V'
            })

        # 5. Detect non-linearity through residuals
        residuals = v_o_simulated - predictions
        residual_std = np.std(residuals)
        if residual_std > 0.15:
            anomalies.append({
                'type': 'non_linearity',
                'severity': residual_std,
                'message': f'Ridge residual standard deviation: {residual_std:.3f} V; not by itself proof of nonlinearity.'
            })

        return anomalies

    def evaluate_circuit_performance(self, regression_results):
        """
        Comprehensive performance evaluation
        """
        ols_coef = regression_results['ols']['coefficients']
        constrained_gain = regression_results['constrained']['gain']

        print("\n--- CIRCUIT PERFORMANCE EVALUATION ---")

        # Gain accuracy assessment
        gain_error_ols = abs(ols_coef[1] - self.theoretical_gain) / abs(self.theoretical_gain)
        gain_error_constrained = abs(constrained_gain - self.theoretical_gain) / abs(self.theoretical_gain)

        print(f"Theoretical Differential Gain: {self.theoretical_gain}")
        print(f"OLS coefficient of v_i2: {ols_coef[1]:.3f} (Error: {gain_error_ols * 100:.1f}%)")
        print(f"Measured Gain (Constrained): {constrained_gain:.3f} (Error: {gain_error_constrained * 100:.1f}%)")

        # Symmetry assessment
        symmetry_ratio = abs(ols_coef[0] + ols_coef[1]) / (abs(ols_coef[0]) + abs(ols_coef[1]) + 1e-8)
        print(f"Coefficient Symmetry (|a+b|/(|a|+|b|)): {symmetry_ratio:.3f} (ideal: 0)")

        # This is a descriptive comparison of four synthetic points, not independent validation.
        print("Note: only four points are available; this comparison is not independent validation.")
        # Performance grading
        max_gain_error = min(gain_error_ols, gain_error_constrained)
        if max_gain_error < 0.1 and symmetry_ratio < 0.1:
            print("Performance: Values agree closely with the theoretical calculation")
        elif max_gain_error < 0.2 and symmetry_ratio < 0.2:
            print("Performance: Values show minor deviations from the theoretical calculation")
        else:
            print("Performance: Values show deviations from the theoretical calculation")

        return {
            'gain_error': min(gain_error_ols, gain_error_constrained),
            'symmetry_deviation': symmetry_ratio,
            'dc_offset': ols_coef[2]
        }

    def analyze_differential_amplifier(self):
        """
        Complete analysis pipeline for differential amplifier
        """
        print("=" * 70)
        print("DIFFERENTIAL AMPLIFIER SYSTEM IDENTIFICATION ANALYSIS")
        print("=" * 70)

        # Load synthetic data
        v_i1, v_i2, v_o_simulated = self.load_synthetic_data()

        print("Synthetic Data Summary:")
        print(f"v_i1: {v_i1}")
        print(f"v_i2: {v_i2}")
        print(f"v_o simulated: {v_o_simulated}")

        # Perform multiple linear regression
        regression_results = self.perform_multiple_linear_regression(v_i1, v_i2, v_o_simulated)

        print("\n--- REGRESSION ANALYSIS RESULTS ---")
        ols_coef = regression_results['ols']['coefficients']
        ridge_coef = regression_results['ridge']['coefficients']
        constrained = regression_results['constrained']

        print("Multiple Linear Regression (v_o = a·v_i1 + b·v_i2 + c):")
        print(f"OLS Coefficients:    a = {ols_coef[0]:.3f}, b = {ols_coef[1]:.3f}, c = {ols_coef[2]:.3f}")
        print(f"Ridge Coefficients:  a = {ridge_coef[0]:.3f}, b = {ridge_coef[1]:.3f}, c = {ridge_coef[2]:.3f}")
        print(f"Constrained Model:   Gain = {constrained['gain']:.3f}, Offset = {constrained['offset']:.3f}")

        print("\nCross-validation R² not reported: four points give test folds with fewer than two samples.")

        # Anomaly detection
        anomalies = self.detect_anomalies(v_i1, v_i2, v_o_simulated, regression_results)

        if anomalies:
            print(f"\n--- ANOMALY DETECTION REPORT ---")
            for anomaly in anomalies:
                print(f"⚠️  {anomaly['message']}")
        else:
            print(f"\n✅ No significant anomalies detected")

        # Performance evaluation
        performance_metrics = self.evaluate_circuit_performance(regression_results)

        return v_i1, v_i2, v_o_simulated, regression_results, anomalies, performance_metrics

    def plot_comprehensive_analysis(self, v_i1, v_i2, v_o_simulated, regression_results):
        """
        Create comprehensive visualization of analysis results
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        predictions = regression_results['ridge']['predictions']
        differential_input = v_i2 - v_i1

        # Plot 1: Input-Output Characteristics
        ax1.scatter(differential_input, v_o_simulated, color='blue', s=80, label='Synthetic data')
        ax1.scatter(differential_input, predictions, color='red', s=60, label='Ridge model predictions')

        # Plot ideal line
        x_ideal = np.linspace(min(differential_input), max(differential_input), 100)
        y_ideal = self.theoretical_gain * x_ideal
        ax1.plot(x_ideal, y_ideal, 'k--', alpha=0.7, label=f'Ideal (gain={self.theoretical_gain})')

        ax1.set_xlabel('Differential input ($v_{i2} - v_{i1}$) (V)', fontsize=16)
        ax1.set_ylabel('Output voltage $v_o$ (V)', fontsize=16)
        ax1.set_title('Differential amplifier characteristics', fontsize=18)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', which='major', labelsize=14)

        # Plot 2: Residual Analysis
        residuals = v_o_simulated - predictions
        ax2.scatter(predictions, residuals, color='red', s=80)
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.set_xlabel('Predicted values (V)', fontsize=16)
        ax2.set_ylabel('Residuals (V)', fontsize=16)
        ax2.set_title('Ridge residual analysis', fontsize=18)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', which='major', labelsize=14)

        # Add residual statistics
        ax2.text(0.05, 0.95, f'Residual Std: {np.std(residuals):.3f} V\n'
                             f'Mean: {np.mean(residuals):.3f} V',
                 transform=ax2.transAxes, verticalalignment='top', fontsize=12,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 3: Common-Mode Analysis
        common_mode = (v_i1 + v_i2) / 2
        ax3.scatter(common_mode, v_o_simulated, color='green', s=80)
        ax3.set_xlabel('Common-mode voltage (V)', fontsize=16)
        ax3.set_ylabel('Output voltage $v_o$ (V)', fontsize=16)
        ax3.set_title('Common-mode and output (mixed inputs)', fontsize=16)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', which='major', labelsize=14)

        # Plot 4: Coefficient Comparison
        methods = ['Theoretical', 'OLS', 'Ridge', 'Constrained']
        gains = [self.theoretical_gain,
                 regression_results['ols']['coefficients'][1],
                 regression_results['ridge']['coefficients'][1],
                 regression_results['constrained']['gain']]

        bars = ax4.bar(methods, gains, color=['gray', 'blue', 'orange', 'green'], alpha=0.7)
        ax4.set_ylabel('Coefficient of $v_{i2}$', fontsize=16)
        ax4.set_title('Model comparison: $v_{i2}$ coefficient', fontsize=18)
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis='both', which='major', labelsize=14)

        # Add value labels on bars
        for i, v in enumerate(gains):
            ax4.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=12)

        plt.tight_layout()
        output_path = Path(__file__).resolve().with_name('Lab3-2_figure1.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {output_path}')

    def plot_transfer_function_comparison(self, v_i1, v_i2, v_o_simulated, regression_results):
        """
        Additional plot: Theoretical vs Synthetic transfer function comparison
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Calculate theoretical values
        v_o_theoretical = self.theoretical_gain * (v_i2 - v_i1)
        v_o_ols = regression_results['ols']['predictions']
        v_o_constrained = regression_results['constrained']['predictions']

        # Create index for x-axis
        x_index = np.arange(len(v_i1))

        width = 0.25
        ax.bar(x_index - width, v_o_theoretical, width, label='Theoretical', alpha=0.7)
        ax.bar(x_index, v_o_simulated, width, label='Synthetic', alpha=0.7)
        ax.bar(x_index + width, v_o_ols, width, label='OLS model', alpha=0.7)

        ax.set_xlabel('Measurement point', fontsize=16)
        ax.set_ylabel('Output voltage $v_o$ (V)', fontsize=16)
        ax.set_title('Theoretical vs experimental transfer function comparison', fontsize=18)
        ax.set_xticks(x_index)
        ax.set_xticklabels([f'Point {i + 1}' for i in range(len(v_i1))])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        output_path = Path(__file__).resolve().with_name('Lab3-2_figure2.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Figure saved to: {output_path}')


def main():
    """
    Main execution function for differential amplifier analysis
    """
    # Figure 7: Rf/Rin = 100 kΩ / 10 kΩ = 10
    analyzer = DifferentialAmplifierAnalyzer(theoretical_gain=10.0)

    # Perform complete analysis
    v_i1, v_i2, v_o_simulated, regression_results, anomalies, performance = \
        analyzer.analyze_differential_amplifier()

    # Generate comprehensive plots
    analyzer.plot_comprehensive_analysis(v_i1, v_i2, v_o_simulated, regression_results)
    analyzer.plot_transfer_function_comparison(v_i1, v_i2, v_o_simulated, regression_results)
    plt.show()  # Both figures are opened together after saving.

    # Final assessment report
    print("\n" + "=" * 70)
    print("FINAL SYSTEM IDENTIFICATION REPORT")
    print("=" * 70)

    ols_coef = regression_results['ols']['coefficients']
    print(f"Identified Transfer Function: v_o = {ols_coef[0]:.3f}·v_i1 + {ols_coef[1]:.3f}·v_i2 + {ols_coef[2]:.3f}")
    print(
        f"Theoretical Expectation: v_o = {-analyzer.theoretical_gain:.3f}·v_i1 + {analyzer.theoretical_gain:.3f}·v_i2 + 0")

    print("\nFour synthetic points are insufficient for independent validation of the model.")

    # Print detailed data table
    print(f"\n--- DETAILED EXPERIMENTAL DATA ---")
    print(f"{'Point':<8} {'v_i1 (V)':<12} {'v_i2 (V)':<12} {'v_o sim (V)':<15} {'v_o theo (V)':<15} {'Error (%)':<12}")
    print("-" * 75)
    for i in range(len(v_i1)):
        v_o_theo = analyzer.theoretical_gain * (v_i2[i] - v_i1[i])
        error_pct = f'{abs((v_o_simulated[i] - v_o_theo) / v_o_theo * 100):.2f}' if v_o_theo != 0 else 'N/A'
        print(
            f"{i + 1:<8} {v_i1[i]:<12.2f} {v_i2[i]:<12.2f} {v_o_simulated[i]:<15.3f} {v_o_theo:<15.3f} {error_pct:<12}")


if __name__ == "__main__":
    print("Dataset: synthetic voltages generated by a simple gain model.")
    main()