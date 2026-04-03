import numpy as np
import neurokit2 as nk
from scipy import signal

class ECGSimulator:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        
    def generate_ecg(self, signal_type="Normal Sinus Rhythm", duration=10, noise_level=0.1):
        num_samples = int(duration * self.sampling_rate)
        
        if signal_type == "Normal Sinus Rhythm":
            heart_rate = np.random.randint(60, 90)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level,
                method='ecgsyn'
            )
            
        elif signal_type == "Atrial Fibrillation":
            heart_rate = np.random.randint(100, 150)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level * 2,
                method='ecgsyn'
            )
            ecg = self._add_afib_irregularity(ecg)
            
        elif signal_type == "Bradycardia":
            heart_rate = np.random.randint(40, 55)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level,
                method='ecgsyn'
            )
            
        elif signal_type == "Tachycardia":
            heart_rate = np.random.randint(110, 150)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level,
                method='ecgsyn'
            )
            
        elif signal_type == "High Stress":
            heart_rate = np.random.randint(85, 110)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level * 1.5,
                method='ecgsyn'
            )
            ecg = self._add_stress_variability(ecg)
            
        elif signal_type == "Relaxed":
            heart_rate = np.random.randint(55, 70)
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level * 0.5,
                method='ecgsyn'
            )
            
        else:
            heart_rate = 75
            ecg = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.sampling_rate,
                heart_rate=heart_rate,
                noise=noise_level,
                method='ecgsyn'
            )
        
        ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=self.sampling_rate)
        
        try:
            signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=self.sampling_rate)
            rpeaks = info['ECG_R_Peaks']
        except:
            rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=self.sampling_rate)[1]['ECG_R_Peaks']
        
        if len(rpeaks) > 1:
            rr_intervals = np.diff(rpeaks) / self.sampling_rate
            actual_heart_rate = 60 / np.mean(rr_intervals)
        else:
            actual_heart_rate = heart_rate
        
        return {
            'ecg_signal': ecg_cleaned,
            'rpeaks': rpeaks,
            'sampling_rate': self.sampling_rate,
            'heart_rate': actual_heart_rate,
            'duration': duration,
            'signal_type': signal_type
        }
    
    def _add_afib_irregularity(self, ecg):
        irregularity = np.random.normal(0, 0.15, len(ecg))
        time_warping = signal.resample(irregularity, len(ecg))
        ecg_irregular = ecg + time_warping
        
        p_wave_suppression = np.ones(len(ecg))
        for i in range(0, len(ecg), self.sampling_rate):
            if np.random.random() > 0.3:
                start = max(0, i - int(0.1 * self.sampling_rate))
                end = min(len(ecg), i + int(0.1 * self.sampling_rate))
                p_wave_suppression[start:end] *= 0.3
        
        ecg_irregular *= p_wave_suppression
        return ecg_irregular
    
    def _add_stress_variability(self, ecg):
        low_freq_noise = np.sin(2 * np.pi * 0.1 * np.arange(len(ecg)) / self.sampling_rate)
        high_freq_noise = np.random.normal(0, 0.05, len(ecg))
        
        ecg_stressed = ecg + 0.1 * low_freq_noise + high_freq_noise
        return ecg_stressed
