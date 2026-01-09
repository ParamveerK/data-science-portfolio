import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import scipy


class UVVisProcessor:
    def __init__(self, min_wl, max_wl):
        self.min_wl = min_wl
        self.max_wl = max_wl

    def import_spectrum(self,filepath):
        df = pd.read_csv(filepath,sep='\t',header=1)
        return self.clean_spectrum(df)

    def clean_spectrum(self,df):
        df = df.copy()
        df = df.replace(' ',0)
        df = df.set_index('Wavelength nm.').astype(float)
        return df.loc[self.min_wl:self.max_wl].astype(float)

    def batch_import(self,path):
        f = path + '/'
        files = []
        try:
            files = [file for file in os.listdir(f) if file.split('.')[1]=='txt' and self.extract_file_number(file) < 10]
            files.sort(key=lambda x: self.extract_file_number(x))
            print(f'Found files: {files}')
        except FileNotFoundError:
            print(f'No folder/file was found with the name {f}')
        
        spectra = pd.DataFrame(index=range(self.min_wl,self.max_wl))

        for file in files:
            spectrum = self.import_spectrum(f+file)
            file_num = self.extract_file_number(file)
            if self.is_peaking(spectrum) == False:
                spectra[f'spectrum_{file_num}'] = spectrum.iloc[:,0]
            else:
                print(f'spectrum_{file_num} values are too high, omitting')
        return spectra

    def is_peaking(self, spectrum):
        if spectrum[spectrum > 0.1].iloc[:,0].mean() > 2.5:
            return True
        else:
            return False
    
    @staticmethod
    def extract_file_number(filename):
        return int(filename.split('.')[0].split('_')[1])

class UVVisAnalyser:
    def __init__(self, height=0.02, width=4):
        self.height = height
        self.width = width

    def fit_peaks(self, spectrum):
        peaks, _ = scipy.signal.find_peaks(spectrum.values,width=self.width,height=self.height)
        peak_wl = spectrum.iloc[peaks].index.astype(int).tolist()
        peak_abs = spectrum.iloc[peaks].values.astype(float).tolist()
        return peak_wl, peak_abs

    def find_lambda_max(self, spectrum):
        return int(spectrum.astype(float).idxmax())

    def get_absorbance(self, spectra, wavelength):
        return spectra.loc[wavelength].tolist()

class UVVisVisualiser:
    
    def plot_spectra(self, spectra, show_peaks=False ,peak_analyser=None, num_spectra = 3):
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
                
            
