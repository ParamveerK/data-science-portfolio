# UV-Vis Analysis
Python pipeline for batch processing UV-Vis spectra, reducing manual absorbance extraction from ~20 minutes per 5 spectra to seconds. Built to streamline Beer-Lambert analysis during PhD research.

## Problem
During photostability research, I needed to determine Beer-Lambert coefficients for 8-10 compounds, each requiring analysis of 5 concentration-series spectra:

1. Manually import each UV-Vis spectrum into Excel

2. Plot to identify the peak wavelength
3. Record absorbance at the peak
4. Extract absorbance at the same wavelength across all 5 concentrations
5. Repeat for the next compound

This workflow took ~15-20 minutes per compound (5 spectra) and was prone to transcription errors when manually copying absorbance values between files. Analyzing a full experimental set (40-50 spectra) required 2-3 hours of repetitive data entry.

## Solution
Built an object-oriented Python pipeline with three main components:

### UVVisProcessor

- Batch imports all .txt files from Shimadzu spectrometer output

- Automatically filters to specified wavelength range (e.g., 380-760 nm for visible light)

- Quality control: detects and excludes oversaturated spectra

### UVVisAnalyzer

- Identifies peak wavelengths using scipy signal processing

- Extracts absorbance values at specified wavelengths across all spectra

- Configurable peak detection sensitivity for different compound types

### UVVisVisualizer

- Plots multiple spectra with automatic peak annotation

- Enables quick visual verification before extracting values for Beer-Lambert analysis

The pipeline processes an entire compound set (5 spectra) in seconds rather than 20 minutes, with absorbance values exported directly for linear regression.

## Results

- Reduced analysis time from 2-3 hours to ~10 minutes per experimental set (40-50 spectra)

- Eliminated transcription errors by programmatically extracting values

- Successfully analyzed 8 organic compounds for Beer-Lambert coefficient determination

## Technologies
Python | Pandas | NumPy | SciPy | Matplotlib