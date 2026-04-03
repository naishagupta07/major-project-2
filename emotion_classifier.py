import numpy as np

class EmotionClassifier:
    def __init__(self):
        self.emotions = ['Relaxed', 'Neutral', 'Stressed']
        
    def classify(self, hrv_metrics, heart_rate):
        features = self._extract_emotion_features(hrv_metrics, heart_rate)
        
        emotion, probabilities = self._rule_based_classification(features)
        
        valence, arousal = self._determine_valence_arousal(emotion, features)
        
        return {
            'emotion': emotion,
            'probabilities': probabilities,
            'valence': valence,
            'arousal': arousal,
            'features': features
        }
    
    def _extract_emotion_features(self, hrv_metrics, heart_rate):
        rmssd = hrv_metrics.get('rmssd', 30)
        sdnn = hrv_metrics.get('sdnn', 50)
        lf_hf_ratio = hrv_metrics.get('lf_hf_ratio', 1.5)
        stress_level = hrv_metrics.get('stress_level', 'Moderate')
        
        stress_score = {
            'Low': 0,
            'Moderate': 1,
            'High': 2,
            'Unknown': 1
        }.get(stress_level, 1)
        
        return {
            'heart_rate': heart_rate,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'lf_hf_ratio': lf_hf_ratio,
            'stress_score': stress_score
        }
    
    def _rule_based_classification(self, features):
        hr = features['heart_rate']
        sdnn = features['sdnn']
        stress_score = features['stress_score']

        scores = {'Relaxed': 0, 'Neutral': 0, 'Stressed': 0}

        if sdnn > 50 and stress_score == 0:
          scores['Relaxed'] = 0.7
          scores['Neutral'] = 0.2
          scores['Stressed'] = 0.1

        elif sdnn < 30 or stress_score >= 2 or hr > 95:
           scores['Stressed'] = 0.7
           scores['Neutral'] = 0.2
           scores['Relaxed'] = 0.1

        else:
            scores['Neutral'] = 0.6
            scores['Relaxed'] = 0.2
            scores['Stressed'] = 0.2

        total = sum(scores.values())
        probabilities = {k: v/total for k, v in scores.items()}
        dominant_emotion = max(probabilities.items(), key=lambda x: x[1])[0]

        return dominant_emotion, probabilities

    def _determine_valence_arousal(self, emotion, features):
        emotion_map = emotion_map = {
    'Relaxed': ('Positive', 'Low'),
    'Neutral': ('Neutral', 'Medium'),
    'Stressed': ('Negative', 'High')
}
        
        valence, arousal = emotion_map.get(emotion, ('Neutral', 'Medium'))
        
        hr = features['heart_rate']
        if hr > 100:
            arousal = 'High'
        elif hr < 60:
            arousal = 'Low'
        
        return valence, arousal
