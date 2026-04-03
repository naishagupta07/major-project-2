import numpy as np
from scipy import signal, interpolate

class HRVAnalyzer:
    def __init__(self):
        self.freq_bands = {
            'vlf': (0.003, 0.04),
            'lf': (0.04, 0.15),
            'hf': (0.15, 0.4)
        }
    
    def analyze(self, rpeaks, sampling_rate):
        if len(rpeaks) < 2:
            return self._get_default_metrics()
        
        rr_intervals = self._calculate_rr_intervals(rpeaks, sampling_rate)
        
        if len(rr_intervals) < 2:
            return self._get_default_metrics()
        
        time_domain = self._calculate_time_domain(rr_intervals)
        
        freq_domain = self._calculate_frequency_domain(rr_intervals, sampling_rate)
        
        stress_level = self._assess_stress_level(time_domain, freq_domain)
        ans_balance = self._assess_ans_balance(freq_domain)
        recovery_status = self._assess_recovery(time_domain)
        
        return {
            'rr_intervals': rr_intervals,
            'sdnn': time_domain['sdnn'],
            'rmssd': time_domain['rmssd'],
            'pnn50': time_domain['pnn50'],
            'lf_power': freq_domain['lf_power'],
            'hf_power': freq_domain['hf_power'],
            'lf_hf_ratio': freq_domain['lf_hf_ratio'],
            'stress_level': stress_level,
            'ans_balance': ans_balance,
            'recovery_status': recovery_status
        }
    
    def _calculate_rr_intervals(self, rpeaks, sampling_rate):
        rr_intervals = np.diff(rpeaks) / sampling_rate * 1000
        
        median_rr = np.median(rr_intervals)
        valid_rr = rr_intervals[
            (rr_intervals > median_rr * 0.5) & 
            (rr_intervals < median_rr * 1.5)
        ]
        
        return valid_rr if len(valid_rr) > 0 else rr_intervals
    
    def _calculate_time_domain(self, rr_intervals):
        sdnn = np.std(rr_intervals, ddof=1)
        
        successive_diffs = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diffs ** 2))
        
        nn50 = np.sum(np.abs(successive_diffs) > 50)
        pnn50 = (nn50 / len(successive_diffs)) * 100 if len(successive_diffs) > 0 else 0
        
        return {
            'sdnn': sdnn,
            'rmssd': rmssd,
            'pnn50': pnn50
        }
    
    def _calculate_frequency_domain(self, rr_intervals, sampling_rate):
        if len(rr_intervals) < 10:
            return {
                'lf_power': 0,
                'hf_power': 0,
                'lf_hf_ratio': 1.0
            }
        
        time_rr = np.cumsum(rr_intervals) / 1000.0
        time_rr = np.insert(time_rr, 0, 0)
        
        fs_interp = 4.0
        time_interp = np.arange(0, time_rr[-1], 1/fs_interp)
        
        if len(time_rr) < 2 or len(np.unique(rr_intervals)) < 2:
            return {
                'lf_power': 0,
                'hf_power': 0,
                'lf_hf_ratio': 1.0
            }
        
        interp_func = interpolate.interp1d(
            time_rr, 
            np.append(rr_intervals, rr_intervals[-1]), 
            kind='cubic',
            fill_value='extrapolate'
        )
        rr_interp = interp_func(time_interp)
        
        rr_interp = rr_interp - np.mean(rr_interp)
        
        freqs, psd = signal.welch(
            rr_interp,
            fs=fs_interp,
            nperseg=min(256, len(rr_interp)),
            window='hann'
        )
        
        lf_mask = (freqs >= self.freq_bands['lf'][0]) & (freqs < self.freq_bands['lf'][1])
        hf_mask = (freqs >= self.freq_bands['hf'][0]) & (freqs < self.freq_bands['hf'][1])
        
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if np.any(lf_mask) else 0
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if np.any(hf_mask) else 0
        
        lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 1.0
        
        return {
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio
        }
    
    def _assess_stress_level(self, time_domain, freq_domain):
        stress_score = 0
        
        if time_domain['sdnn'] < 30:
            stress_score += 2
        elif time_domain['sdnn'] < 50:
            stress_score += 1
        
        if time_domain['rmssd'] < 20:
            stress_score += 2
        elif time_domain['rmssd'] < 30:
            stress_score += 1
        
        if freq_domain['lf_hf_ratio'] > 3:
            stress_score += 2
        elif freq_domain['lf_hf_ratio'] > 2:
            stress_score += 1
        
        if stress_score >= 4:
            return "High"
        elif stress_score >= 2:
            return "Moderate"
        else:
            return "Low"
    
    def _assess_ans_balance(self, freq_domain):
        ratio = freq_domain['lf_hf_ratio']
        
        if ratio < 1:
            return "Parasympathetic Dominant"
        elif ratio > 2:
            return "Sympathetic Dominant"
        else:
            return "Balanced"
    
    def _assess_recovery(self, time_domain):
        if time_domain['rmssd'] > 40 and time_domain['pnn50'] > 15:
            return "Good Recovery"
        elif time_domain['rmssd'] > 25:
            return "Moderate Recovery"
        else:
            return "Poor Recovery"
    
    def _get_default_metrics(self):
        return {
            'rr_intervals': np.array([800]),
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0,
            'lf_power': 0,
            'hf_power': 0,
            'lf_hf_ratio': 1.0,
            'stress_level': "Unknown",
            'ans_balance': "Unknown",
            'recovery_status': "Unknown"
        }
