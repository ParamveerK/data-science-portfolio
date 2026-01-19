# PL Stability Analysis

Python pipeline for processing time-series photoluminescence spectra, automating relative emission decay analysis and stretched exponential fitting. Built to streamline photostability assessment during PhD research.

## Problem

During photostability research, I needed to track emission decay over days-to-weeks for organic semiconductor materials, each requiring me to:

- Parse timestamps from metadata to calculate elapsed time

- Integrate emission over specific wavelength ranges for each spectrum

- Normalise to initial intensity

- Fit decay to stretched exponential model

- Repeat for multiple emission bands (e.g., blue vs red emission)

This workflow involved significant manual data wrangling in Excel, scrolling through unwieldy Excel sheets with inconsistent cell referencing. Comparing multiple wavelength ranges or re-analysing with different parameters meant repeating the entire process.

## Solution

Built an object-oriented Python pipeline:

### StabilityExperiment

- Loads and parses spectrometer Excel exports, handling mixed date/time formats

- Extracts spectral data with configurable starting spectrum (to exclude initial equilibration)

- Calculates normalised integrals over user-defined wavelength ranges

- Automatically computes elapsed time from embedded timestamps

## Analysis & Fitting

- Fits emission decay to stretched exponential: A·exp(-(t/τ)^β)

- Exports fitted parameters (amplitude, characteristic time, stretching exponent)

- Generates plots comparing experimental data with fitted curves

## Results

- Reduced analysis time from ~30 minutes to seconds per experiment

- Enabled rapid comparison of multiple emission bands in a single analysis

- Applied to 30 experiments during PhD research

## Technologies

Python | Pandas | NumPy | SciPy | Matplotlib | Seaborn
