# Op-Amp Data Analysis

A Python portfolio example for comparing inverting and differential amplifier transfer characteristics with ideal gain models. The scripts generate synthetic voltages from explicit equations and illustrate fitting, residual analysis, and plotting.

## Stack and workflow

- NumPy and SciPy for numerical analysis.
- scikit-learn for linear, robust, and regularized fits.
- Matplotlib for transfer curves and residual plots.

## Run locally

Use Python 3.8–3.11. The scripts write one inverting-amplifier plot and two differential-amplifier plots beside the source files.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
MPLBACKEND=Agg python inverting_amplifier.py
MPLBACKEND=Agg python differential_amplifier.py
```

## Portfolio contribution

The portfolio work focuses on circuit-data analysis in Python, including gain estimation, residual interpretation, and visualization. This edition uses simulated voltages throughout. Results illustrate the workflow and do not establish real hardware performance.
