"""
Module for analysing photoluminescence stability from time-series emission spectra.

Processes spectral data to extract normalised emission integrals over specified 
wavelength ranges, fits decay to a stretched exponential, and generates plots 
for assessing material photostability.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

class StabilityExperiment:
    """A single photoluminescence stability experiment.
        
        Args:
            filename: Path to the Excel file containing spectral data.
            start_spectrum: Index of first spectrum to include (omits earlier spectra).
            wavelength_ranges: Dict mapping names to (min, max) wavelength tuples
                for integration, e.g. {'green': (500, 550)}.
        
        Example:
            exp = StabilityExperiment(
                'stability_test_1.xls',
                start_spectrum=1,
                wavelength_ranges={'green': (500, 550), 'red': (600, 650)}
            )
            exp.plot_pl()
        """
    def __init__(self,filename, start_spectrum, wavelength_ranges):
        file = pd.read_excel(filename)
        file = file.iloc[1:]
        file = file.set_index(file.columns[0])
        numbered_columns = [column for column in file.columns if type(column) == int]
        self.filename = filename
        self.file = file[numbered_columns]
        self.start_spectrum = start_spectrum
        self.wavelength_ranges = wavelength_ranges
        self.spectra = None
        self.datetimes = None
        self.norm_integrals = None
        self.elapsed_hours = None
        self.popt = None

    def get_spectra(self):
        """Extracts spectral data from file.
        
        Returns:
            None. Sets self.spectra.
        """
        spectra = self.file.loc[380:760]
        spectra.index = spectra.index.astype(float)
        self.spectra = spectra.iloc[:,self.start_spectrum:]
        return 
        
    def get_normalised_integrals(self):
        """Calculates integrals over specified wavelength ranges, normalised to first spectrum.
        
        Returns:
            pd.DataFrame: Normalised integrals for each wavelength range.
        """
        self.norm_integrals = pd.DataFrame()
        self.get_spectra()
        for key, value in self.wavelength_ranges.items():
            integral_range = self.spectra.loc[value[0]:value[1]]
            integrals = integral_range.apply(np.trapz,axis=0)
            column_name = self.filename.split('.')[0] + '_' + key
            self.norm_integrals[column_name] = integrals/integrals.iloc[0]
        return self.norm_integrals
    
    def get_datetimes(self):
        """Parses date and time from file metadata.
        
        Returns:
            pd.Series: Datetime objects for each spectrum.
        """
        self.file.loc['datetime'] = self.file.loc['date'] + ' ' + self.file.loc['time']
        self.datetimes = pd.to_datetime(self.file[self.start_spectrum:].loc['datetime'], format='mixed', dayfirst=True)
        return self.datetimes
    
    def get_elapsed_time(self):
        """Calculates elapsed time from first spectrum in hours.
        
        Returns:
            pd.Series: Elapsed time in hours.
        """
        self.get_datetimes()

        elapsed_time = self.datetimes - self.datetimes.iloc[self.start_spectrum]
        self.elapsed_hours = elapsed_time.dt.total_seconds() / 3600
        return self.elapsed_hours
    
    def get_pl_time(self):
        """Calculates normalised integrals and elapsed time.
        
        Args:
            None
        
        Returns:
            pd.DataFrame: Elapsed time and normalised integrals for each wavelength range.
        """
        self.get_normalised_integrals()
        self.get_elapsed_time()
        
        self.pl_time = self.norm_integrals.copy()
        self.pl_time.insert(0,'elapsed time (h)', self.elapsed_hours)
        return self.pl_time
            
    def plot_pl(self):
        """Plots the relative PL change over time.
    
        Args:
            None
            
        Returns:
            None. Displays a matplotlib line plot.
        """
        self.get_pl_time()
        self.pl_time.plot(x='elapsed time (h)')
    
    def fit_exponential(self, colour):
        """Fits a stretched exponential to the decay data.
        
        Args:
            colour: String to match against column names (e.g. 'green').
        
        Returns:
            np.ndarray: Fitted parameters [A, tau, beta].
        """
        self.get_pl_time()
        colour_col = [col for col in self.pl_time.columns if colour in col][0]
        print(colour_col)
        self.popt, _ = curve_fit(stretched_exponential,self.pl_time['elapsed time (h)'],self.pl_time[colour_col],p0=[1,100,0.5])
        print('Stretched exponential has been fitted!')
        return self.popt
        
    def get_fit_table(self, colour, time_range):
        """Generates fitted curve values over a time range.
        
        Args:
            colour: String to match against column names.
            time_range: Maximum time (hours) for output.
        
        Returns:
            pd.DataFrame: Time points and corresponding fitted values.
        """
        self.fit_exponential(colour)
        output_range = [i for i in range(0,round(time_range))]
        output_table = pd.DataFrame({
        'elapsed_time':output_range,
        'function':[stretched_exponential(i,self.popt[0],self.popt[1],self.popt[2]) for i in output_range]})
        return output_table
    
    def plot_stretched(self, colour, time_range):
        """Plots experimental data with fitted stretched exponential.
    
        Args:
            colour: String to match against column names.
            time_range: Maximum time (hours) for the fitted curve.
        
        Returns:
            None. Displays scatter plot with fitted curve.
        """
        colour_col = [col for col in self.pl_time.columns if colour in col]
        exp_table = self.get_fit_table(colour, time_range)
        plt.scatter(self.pl_time['elapsed time (h)'],self.pl_time[colour_col])
        plt.plot(exp_table['elapsed_time'],exp_table['function'])
        plt.xlabel('Time / h')
        plt.ylabel('Relative PL Change')
        plt.title(f'Stretched Exponential Fit of {self.filename.split(".")[0]}')
        plt.show()




def stretched_exponential(t, A, tau, beta):
    """Stretched exponential decay function.
    
    Args:
        t: Time value(s).
        A: Amplitude.
        tau: Characteristic time constant.
        beta: Stretching exponent (0 < beta <= 1).
    
    Returns:
        float or np.ndarray: A * exp(-(t/tau)^beta)
    """
    return A * np.exp(-(t/tau)** beta)