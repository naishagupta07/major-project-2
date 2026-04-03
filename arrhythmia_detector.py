import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy import signal, stats

class ArrhythmiaDetector:
    def __init__(self):
        self.classes = [
            'Normal Sinus Rhythm',
            'Atrial Fibrillation',
            'Bradycardia',
            'Tachycardia',
            'Ventricular Tachycardia'
        ]
        self.model_rf = self._create_rf_model()
        self.model_svm = self._create_svm_model()
    
    def _create_rf_model(self):
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        X_train, y_train = self._generate_synthetic_training_data()
        model.fit(X_train, y_train)
        return model
    
    def _create_svm_model(self):
        model = SVC(
            kernel='rbf',
            C=1.0,
            probability=True,
            random_state=42
        )
        
        X_train, y_train = self._generate_synthetic_training_data()
        model.fit(X_train, y_train)
        return model
    
    def _generate_synthetic_training_data(self, n_samples=500):
        X = []
        y = []
        
        for class_idx, class_name in enumerate(self.classes):
            for _ in range(n_samples // len(self.classes)):
                features = self._generate_synthetic_features(class_name)
                X.append(features)
                y.append(class_idx)
        
        return np.array(X), np.array(y)
    
    def _generate_synthetic_features(self, class_name):
        if class_name == 'Normal Sinus Rhythm':
            mean_rr = np.random.normal(850, 50)
            std_rr = np.random.normal(50, 10)
            hr = 60000 / mean_rr
            hr_var = np.random.normal(0.05, 0.01)
            qrs = np.random.normal(90, 10)
            energy = np.random.normal(1.0, 0.1)
            
        elif class_name == 'Atrial Fibrillation':
            mean_rr = np.random.normal(600, 100)
            std_rr = np.random.normal(150, 30)
            hr = 60000 / mean_rr
            hr_var = np.random.normal(0.25, 0.05)
            qrs = np.random.normal(85, 15)
            energy = np.random.normal(0.8, 0.15)
            
        elif class_name == 'Bradycardia':
            mean_rr = np.random.normal(1200, 100)
            std_rr = np.random.normal(40, 10)
            hr = 60000 / mean_rr
            hr_var = np.random.normal(0.04, 0.01)
            qrs = np.random.normal(95, 10)
            energy = np.random.normal(0.9, 0.1)
            
        elif class_name == 'Tachycardia':
            mean_rr = np.random.normal(500, 50)
            std_rr = np.random.normal(30, 10)
            hr = 60000 / mean_rr
            hr_var = np.random.normal(0.06, 0.02)
            qrs = np.random.normal(85, 10)
            energy = np.random.normal(1.1, 0.1)
            
        else:
            mean_rr = np.random.normal(400, 50)
            std_rr = np.random.normal(60, 15)
            hr = 60000 / mean_rr
            hr_var = np.random.normal(0.15, 0.03)
            qrs = np.random.normal(120, 20)
            energy = np.random.normal(1.3, 0.2)
        
        peak_count = int(np.random.normal(hr / 60 * 10, 2))
        
        return [mean_rr, std_rr, hr_var, qrs, energy, peak_count]
    
    def detect(self, ecg_signal, sampling_rate, model_type="Lightweight RF"):
        features = self._extract_features(ecg_signal, sampling_rate)
        
        if model_type == "Lightweight RF":
            model = self.model_rf
        elif model_type == "Optimized SVM":
            model = self.model_svm
        else:
            model = self.model_rf
        
        feature_vector = np.array([[
            features['mean_rr'],
            features['std_rr'],
            features['hr_variability'],
            features['qrs_duration'],
            features['signal_energy'],
            features['peak_count']
        ]])
        
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        classification = self.classes[prediction]
        confidence = probabilities[prediction]
        
        prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        
        return {
            'classification': classification,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'features': features
        }
    
    def _extract_features(self, ecg_signal, sampling_rate):
        try:
            peaks, _ = signal.find_peaks(ecg_signal, distance=sampling_rate*0.4, prominence=0.3)
        except:
            peaks = np.array([])
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / sampling_rate * 1000
            mean_rr = np.mean(rr_intervals)
            std_rr = np.std(rr_intervals)
            hr = 60000 / mean_rr if mean_rr > 0 else 75
            hr_variability = std_rr / mean_rr if mean_rr > 0 else 0.05
        else:
            mean_rr = 800
            std_rr = 50
            hr = 75
            hr_variability = 0.05
        
        qrs_duration = self._estimate_qrs_duration(ecg_signal, sampling_rate)
        
        signal_energy = np.sum(ecg_signal ** 2) / len(ecg_signal)
        
        peak_count = len(peaks)
        
        return {
            'mean_rr': mean_rr,
            'std_rr': std_rr,
            'hr_variability': hr_variability,
            'qrs_duration': qrs_duration,
            'signal_energy': signal_energy,
            'peak_count': peak_count
        }
    
    def _estimate_qrs_duration(self, ecg_signal, sampling_rate):
        try:
            derivative = np.gradient(ecg_signal)
            derivative_squared = derivative ** 2
            
            threshold = np.mean(derivative_squared) + 2 * np.std(derivative_squared)
            qrs_regions = derivative_squared > threshold
            
            qrs_widths = []
            in_qrs = False
            qrs_start = 0
            
            for i, is_qrs in enumerate(qrs_regions):
                if is_qrs and not in_qrs:
                    qrs_start = i
                    in_qrs = True
                elif not is_qrs and in_qrs:
                    qrs_widths.append(i - qrs_start)
                    in_qrs = False
            
            if qrs_widths:
                avg_qrs_samples = np.median(qrs_widths)
                qrs_duration_ms = (avg_qrs_samples / sampling_rate) * 1000
                return qrs_duration_ms
            else:
                return 90.0
        except:
            return 90.0
