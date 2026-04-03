import wfdb
import numpy as np

def load_ecg_record(record_name, data_dir='mit-data'):
    """
    Loads an ECG record and its annotations from the MIT-BIH database.
    """
    record_path = f'{data_dir}/{record_name}'
    
    try:
        # Read the signal data
        # p_signal = physical signal (the actual ECG data)
        # fs = sampling frequency (e.g., 360 Hz)
        record = wfdb.rdrecord(record_path)
        
        # We'll just use the first channel (MLII) for simplicity
        ecg_signal = record.p_signal[:, 0]
        
        # Read the "truth" annotations (the actual diagnosis)
        # atr = annotation file
        annotation = wfdb.rdann(record_path, 'atr')
        
        return {
            'ecg_signal': ecg_signal,
            'sampling_rate': record.fs,
            'true_diagnosis_symbols': annotation.symbol,
            'true_diagnosis_times': annotation.sample,
            'record_name': record_name
        }
        
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        return None