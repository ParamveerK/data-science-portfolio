import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy


class UVVisProcessor:
    def __init__(self, min_wl, max_wl):
        """
        :param self: UVVisProcessor instance
        :param min_wl: int
                Minimum wavelength range for analysis
        :param max_wl: int
                Maximum wavelength range for analysis
        """
        self.min_wl = min_wl # Minimum wavelength range for analysis
        self.max_wl = max_wl # Maximum wavelength range for analysis

    def import_spectrum(self,filepath):
        """
        Import raw spectrum data from Shimadzu spectrometer and clean it.
        :param filepath: str
            Path to raw spectrum file
        :return: Cleaned DataFrame with wavelength as index
        """
        df = pd.read_csv(filepath,sep='\t',header=1) # Read raw spectrum data, tab-separated with 1 header row
        return self.clean_spectrum(df) # Clean and format the spectrum data

    def clean_spectrum(self,df):
        """
        Clean and format raw spectrum data by handling empty values, setting wavelength
        index, and filtering to specified range.
        
        :param df: DataFrame
            Raw spectrum from Shimadzu spectrometer with 'Wavelength nm.' and 'Abs.' columns

        :return: Cleaned DataFrame with wavelenght as index, filtered to min_wl and max_wl
        """

        df = df.copy() # Avoid modifying original DataFrame
        df = df.replace(' ',0) # Replace empty strings with 0
        df = df.set_index('Wavelength nm.').astype(float) # Set wavelength as index and convert to float
        return df.loc[self.min_wl:self.max_wl].astype(float) # Filter to specified wavelength range

    def batch_import(self,path):
        """
        Docstring for batch_import
        
        :param self: UVVisProcessor instance
        :param path: str
                Path to directory containing raw spectrum files
        """
        f = path + '/' # Ensure path ends with '/'
        files = [] # Initialize empty list to hold valid files
        try:
            files = [file for file in os.listdir(f) if file.split('.')[1]=='txt' and self.extract_file_number(file) < 10] # List all .txt files in directory with file number < 10
            files.sort(key=lambda x: self.extract_file_number(x)) # Sort files by extracted file number
            print(f'Found files: {files}')
        except FileNotFoundError:
            print(f'No folder/file was found with the name {f}')
        
        spectra = pd.DataFrame(index=range(self.min_wl,self.max_wl)) # Initialize empty DataFrame with wavelength index

        for file in files: # Loop through each file
            spectrum = self.import_spectrum(f+file) # Import and clean spectrum
            file_num = self.extract_file_number(file) # Extract file number for naming
            if self.is_peaking(spectrum) == False: # Check if spectrum is valid (not peaking too high)
                spectra[f'spectrum_{file_num}'] = spectrum.iloc[:,0] # Add spectrum to DataFrame
            else:
                print(f'spectrum_{file_num} values are too high, omitting') # Omit spectrum if peaking too high
        return spectra # Return compiled spectra DataFrame

    def is_peaking(self, spectrum):
        """
        Check if spectrum has excessive peak heights.
        
        :param self: UVVisProcessor instance
        :param spectrum: DataFrame
            Spectrum to check for excessive peak heights
        """
        if spectrum[spectrum > 0.1].iloc[:,0].mean() > 2.5: # Check if mean absorbance above 0.1 is greater than 2.5
            return True
        else:
            return False
    
    @staticmethod
    def extract_file_number(filename): 
        """
        Extract file number from filename.

        :param filename: str
            Name of the file to extract the number from
        """
        return int(filename.split('.')[0].split('_')[1]) # Extract file number from filename

class UVVisAnalyser:
    def __init__(self, height=0.02, width=4):
        """
        Docstring for __init__
        
        :param self: UVVisAnalyser instance
        :param height: float
                Minimum height of peaks to detect
        :param width: int
                Minimum width of peaks to detect
        """
        self.height = height # Minimum height of peaks to detect
        self.width = width # Minimum width of peaks to detect

    def fit_peaks(self, spectrum):
        """
        Fit peaks to a given spectrum using scipy's find_peaks method.

        :param self: UVVisAnalyser instance
        :param spectrum: DataFrame
            Spectrum to fit peaks to
        """
        peaks, _ = scipy.signal.find_peaks(spectrum.values,width=self.width,height=self.height) # Detect peaks in spectrum
        peak_wl = spectrum.iloc[peaks].index.astype(int).tolist() # Get wavelengths of detected peaks
        peak_abs = spectrum.iloc[peaks].values.astype(float).tolist() # Get absorbance values of detected peaks
        return peak_wl, peak_abs

    def find_lambda_max(self, spectrum):
        """
        Find the wavelength of maximum absorbance in a spectrum.

        :param self: UVVisAnalyser instance
        :param spectrum: DataFrame
            Spectrum to find lambda max for
        """
        return int(spectrum.astype(float).idxmax()) # Find wavelength of maximum absorbance

    def get_absorbance(self, spectra, wavelength):
        return spectra.loc[wavelength].tolist() # Get absorbance values at specified wavelength for all spectra

class UVVisVisualiser:
    """Visualise UV-Vis spectra"""
    def plot_spectra(self, spectra, show_peaks=False ,peak_analyser=None, num_spectra = 3):
        """
        Plot UV-Vis spectra with optional peak annotation.
        :param self: UVVisVisualiser instance
        :param spectra: DataFrame
            DataFrame containing UV-Vis spectra to plot
        :param show_peaks: bool
            Whether to annotate detected peaks on the plot
        :param peak_analyser: UVVisAnalyser
            UVVisAnalyser instance for peak detection
        :param num_spectra: int
            Number of spectra to annotate with peaks
        """
        fig, ax = plt.subplots(figsize=(10,6))
        ax.plot(spectra)
        ax.legend([column for column in spectra.columns])
        if show_peaks and peak_analyser:
            for spectrum in spectra.columns[:num_spectra]:
                peaks, absorbances = peak_analyser.fit_peaks(spectra[spectrum])
                ax.scatter(peaks, absorbances)
                for peak, absorbance in zip(peaks, absorbances):
                    plt.text(peak, absorbance, f'{peak:.0f}', ha='center',va='bottom')
        plt.show()
                
            
