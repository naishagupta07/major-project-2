import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import json
import os  # <-- ADDED THIS IMPORT
import neurokit2 as nk

import base64

def add_bg_local(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()

    css = f"""
    <style>

    [data-testid="stAppViewContainer"] {{
        background-image: 
            linear-gradient(rgba(20,20,20,0.55), rgba(20,20,20,0.55)),
            url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid="stSidebar"] {{
        background-color: rgba(0, 0, 0, 0.6);
    }}

    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0.3);
    }}

    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

from data_loader import *
from hrv_analyzer import HRVAnalyzer
from arrhythmia_detector import ArrhythmiaDetector
from emotion_classifier import EmotionClassifier
from edge_ai_metrics import EdgeAIMetrics

st.set_page_config(
    page_title="Edge AI ECG Analysis System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
        color: #c62828; /* DARK TEXT */
    }
    .alert-warning {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        color: #e65100; /* DARK TEXT */
    }
    .alert-normal {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
        color: #1b5e20; /* DARK TEXT */
    }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'hrv_analyzer' not in st.session_state:
        st.session_state.hrv_analyzer = HRVAnalyzer()
    if 'arrhythmia_detector' not in st.session_state:
        st.session_state.arrhythmia_detector = ArrhythmiaDetector()
    if 'emotion_classifier' not in st.session_state:
        st.session_state.emotion_classifier = EmotionClassifier()
    if 'edge_metrics' not in st.session_state:
        st.session_state.edge_metrics = EdgeAIMetrics()
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'monitoring' not in st.session_state:
        st.session_state.monitoring = False

def main():
    add_bg_local("background.jpg")
    initialize_session_state()
    
    st.markdown('<p class="main-header">🫀 Edge AI-Based Real-Time ECG Analysis System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Stress, Emotion & Arrhythmia Detection | AI + Healthcare + Edge Computing</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
        st.title("Control Panel")
        
        st.subheader("Signal Configuration")
        
        # --- FIXED DYNAMIC SELECTBOX CODE ---
        mit_data_path = 'mit-data' # The name of your data folder
        available_records = []
        if os.path.exists(mit_data_path):
            for filename in os.listdir(mit_data_path):
                if filename.endswith('.hea'): # We only need the .hea files to identify records
                    # Extract the record number (e.g., '100' from '100.hea')
                    record_num = filename.split('.')[0]
                    available_records.append(record_num)
            # Sort records numerically for better user experience
            if available_records:
                available_records = sorted(list(set(available_records)), key=int)
            else:
                available_records = ["No records found"]
        else:
            st.error(f"Error: The '{mit_data_path}' folder was not found.")
            available_records = ["No records found"] # Fallback if folder is missing

        record_name = st.selectbox(
            "Select Patient Record",
            available_records # Use the dynamically generated list
        )
        # --- END OF FIX ---
        
        st.subheader("Edge AI Settings")
        model_type = st.selectbox("Model Type", ["Lightweight RF", "Quantized NN"])
        edge_device = st.selectbox("Target Device", ["ARM Cortex-M4", "Raspberry Pi"])
        
        st.divider()
        
        if st.button("🔄 Generate & Analyze", type="primary", use_container_width=True):
            analyze_ecg(record_name, model_type, edge_device)
        
        if st.button("📊 View Analytics Dashboard"):
            st.session_state.monitoring = True
        
        st.divider()
        st.caption("💡 **Project Features:**")
        st.caption("✓ Real-time ECG Simulation")
        st.caption("✓ Arrhythmia Detection")
        st.caption("✓ HRV-based Stress Analysis")
        st.caption("✓ HRV-based Emotional State Estimation")
        st.caption("✓ Edge AI Optimization")
    
    tabs = st.tabs(["📈 Live Monitoring", "📉 HRV Analysis", "🧠 AI Diagnostics", "📋 Historical Data", "⚙️ Edge AI Metrics"])
    
    with tabs[0]:
        render_live_monitoring()
    
    with tabs[1]:
        render_hrv_analysis()
    
    with tabs[2]:
        render_ai_diagnostics()
    
    with tabs[3]:
        render_historical_data()
    
    with tabs[4]:
        render_edge_metrics()

def analyze_ecg(record_name, model_type, edge_device):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    status_text.text(f"🔄 Loading Patient Record: {record_name}...")
    progress_bar.progress(10)
    
    loaded_data = load_ecg_record(record_name)
    
    if loaded_data is None:
        st.error(f"Could not load record {record_name}. Check the file path in 'mit-data'.")
        progress_bar.empty()
        status_text.empty()
        return
    
    ecg_data = {
        'ecg_signal': loaded_data['ecg_signal'],
        'sampling_rate': loaded_data['sampling_rate'],
        'rpeaks': None, 
        'heart_rate': 0,
    }
    
    status_text.text("🔍 Finding R-peaks...")
    progress_bar.progress(30)
    try:
        ecg_cleaned = nk.ecg_clean(ecg_data['ecg_signal'], sampling_rate=ecg_data['sampling_rate'])
        _, rpeaks_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=ecg_data['sampling_rate'])
        ecg_data['rpeaks'] = rpeaks_info['ECG_R_Peaks']
    except Exception as e:
        st.warning(f"Could not find R-peaks: {e}")
        ecg_data['rpeaks'] = np.array([])
    
    status_text.text("🔍 Detecting arrhythmias...")
    progress_bar.progress(40)
    time.sleep(0.3)
    
    arrhythmia_result = st.session_state.arrhythmia_detector.detect(
        ecg_data['ecg_signal'],
        ecg_data['sampling_rate'],
        model_type=model_type
    )
    
    status_text.text("💓 Analyzing HRV...")
    progress_bar.progress(60)
    time.sleep(0.3)
    
    hrv_result = st.session_state.hrv_analyzer.analyze(
        ecg_data['rpeaks'],
        ecg_data['sampling_rate']
    )
    
    if 'rr_intervals' in hrv_result and len(hrv_result['rr_intervals']) > 0:
        mean_rr_ms = np.mean(hrv_result['rr_intervals'])
        actual_heart_rate = 60000 / mean_rr_ms if mean_rr_ms > 0 else 0
    else:
        duration_sec = len(ecg_data['ecg_signal']) / ecg_data['sampling_rate']
        actual_heart_rate = (len(ecg_data['rpeaks']) / duration_sec) * 60 if duration_sec > 0 else 0
    
    ecg_data['heart_rate'] = actual_heart_rate
    
    status_text.text("😌 Classifying emotion...")
    progress_bar.progress(80)
    time.sleep(0.3)
    
    emotion_result = st.session_state.emotion_classifier.classify(
        hrv_metrics=hrv_result,
        heart_rate=ecg_data['heart_rate']
    )
    
    # --- LOGICAL OVERRIDE ---
    critical_arrhythmias = ['Atrial Fibrillation', 'Ventricular Tachycardia']

    if arrhythmia_result['classification'] in critical_arrhythmias:
     hrv_result['stress_level'] = "High"
     emotion_result['emotion'] = "Unreliable"
    else:
     emotion_result['emotion'] = emotion_result['emotion']
    # --- END OF OVERRIDE ---

    status_text.text("⚡ Computing edge metrics...")
    progress_bar.progress(90)
    
    edge_result = st.session_state.edge_metrics.compute_metrics(
        model_type=model_type,
        edge_device=edge_device,
        signal_length=len(ecg_data['ecg_signal'])
    )
    
    result = {
        'timestamp': datetime.now(),
        'signal_type': record_name,
        'ecg_data': ecg_data,
        'arrhythmia': arrhythmia_result,
        'hrv': hrv_result,
        'emotion': emotion_result,
        'edge_metrics': edge_result,
        'model_type': model_type,
        'edge_device': edge_device
    }
    
    st.session_state.history.append(result)
    st.session_state.current_result = result
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()

def render_live_monitoring():
    if 'current_result' not in st.session_state:
        st.info("👈 Click 'Generate & Analyze' in the sidebar to start monitoring")
        return
    
    result = st.session_state.current_result
    
    col1, col2, col3, col4 = st.columns([1,2,1,1]) # <-- REVERTED TO 4 EQUAL COLUMNS
    
    with col1:
        st.metric("❤️ Heart Rate", f"{result['ecg_data']['heart_rate']:.0f} BPM")
    
    with col2:
        arrhythmia_type = result['arrhythmia']['classification']
        # Shorten for display
        if arrhythmia_type == "Normal Sinus Rhythm":
            arrhythmia_type = "Normal Sinus"
        
        confidence = result['arrhythmia']['confidence'] * 100
        st.metric("🔍 Arrhythmia", arrhythmia_type, f"{confidence:.1f}% confidence")
    
    with col3:
        stress_level = result['hrv']['stress_level']
        st.metric("😰 Stress Level", stress_level)
    
    with col4:
        emotion = result['emotion']['emotion']
        st.metric("😊 Emotional State (HRV-based)", emotion)
        st.caption("⚠ Approximate — may be unreliable in abnormal heart rhythms")
        
        if result['arrhythmia']['classification'] != "Normal Sinus Rhythm":
         st.warning("⚠ Emotional state estimation may be unreliable due to abnormal ECG pattern")
    
    st.divider()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('ECG Waveform (Lead I)', 'R-Peak Detection'),
        specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
    )
    
    time_axis = np.arange(len(result['ecg_data']['ecg_signal'])) / result['ecg_data']['sampling_rate']
    
    #ecg waveform
    fig.add_trace(
        go.Scatter(x=time_axis, y=result['ecg_data']['ecg_signal'], 
                   mode='lines', name='ECG Signal', line=dict(color='#1f77b4', width=1)),
        row=1, col=1
    )
    
    #r-peak detection
    rpeaks = result['ecg_data']['rpeaks']
    if rpeaks is not None and len(rpeaks) > 0:
        rpeak_times = rpeaks / result['ecg_data']['sampling_rate']
        valid_rpeaks = rpeaks[rpeaks < len(result['ecg_data']['ecg_signal'])]
        rpeak_values = result['ecg_data']['ecg_signal'][valid_rpeaks]
        
        fig.add_trace(
            go.Scatter(x=rpeak_times, y=rpeak_values,
                       mode='markers', name='R-Peaks', marker=dict(color='red', size=8, symbol='x')),
            row=1, col=2
        )
        
        
        
    
        
    hr_trend = np.full(len(time_axis), result['ecg_data']['heart_rate'])
    
    
    
    if 'rr_intervals' in result['hrv'] and len(result['hrv']['rr_intervals']) > 1:
        rr_n = result['hrv']['rr_intervals'][:-1]
        rr_n1 = result['hrv']['rr_intervals'][1:]
        
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    
    
    
    fig.update_yaxes(title_text="Amplitude (mV)", row=1, col=1)
    fig.update_yaxes(title_text="BPM", row=1, col=2)
    
    
    
    # --- FIXED LEGEND ---
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🚨 Alert Status")
    
    if result['arrhythmia']['classification'] in ['Atrial Fibrillation', 'Ventricular Tachycardia']:
        st.markdown(f"""
        <div class="alert-critical">
        <strong>⚠️ CRITICAL ALERT:</strong> {result['arrhythmia']['classification']} detected with {result['arrhythmia']['confidence']*100:.1f}% confidence.
        Immediate medical attention recommended.
        </div>
        """, unsafe_allow_html=True)
    elif result['arrhythmia']['classification'] in ['Bradycardia', 'Tachycardia']:
        st.markdown(f"""
        <div class="alert-warning">
        <strong>⚠️ WARNING:</strong> {result['arrhythmia']['classification']} detected. Monitor patient closely.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-normal">
        <strong>✅ NORMAL:</strong> No critical arrhythmias detected. Vital signs within normal range.
        </div>
        """, unsafe_allow_html=True)

def render_hrv_analysis():
    if 'current_result' not in st.session_state:
        st.info("👈 Generate an ECG signal first to see HRV analysis")
        return
    
    result = st.session_state.current_result
    hrv = result['hrv']
    
    st.subheader("📊 Heart Rate Variability Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Time Domain")
        st.metric("SDNN (ms)", f"{hrv['sdnn']:.2f}", help="Standard deviation of NN intervals")
        
        
    
    
        
        
        
    
    with col3:
        
        st.metric("Stress Level", hrv['stress_level'])
        
        
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RR Interval Distribution")
        fig = go.Figure()
        if 'rr_intervals' in hrv and len(hrv['rr_intervals']) > 0:
            fig.add_trace(go.Histogram(
                x=hrv['rr_intervals'],
                nbinsx=30,
                marker_color='#1f77b4',
                name='RR Intervals'
            ))
        fig.update_layout(
            xaxis_title="RR Interval (ms)",
            yaxis_title="Frequency",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Stress Level Interpretation")
        
        stress_data = {
            'Low': 0,
            'Moderate': 0,
            'High': 0,
            'Unknown': 0
        }
        if hrv['stress_level'] in stress_data:
            stress_data[hrv['stress_level']] = 100
        else:
             stress_data['Unknown'] = 100
        
        fig = go.Figure(go.Bar(
            x=list(stress_data.keys()),
            y=list(stress_data.values()),
            marker_color=['#4caf50', '#ff9800', '#f44336', '#9e9e9e']
        ))
        fig.update_layout(
            yaxis_title="Current Level (%)",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **HRV Clinical Significance:**
    - **High HRV**: Better cardiovascular fitness, good stress adaptation
    - **Low HRV**: Potential stress, fatigue, or cardiovascular issues
    
    """)

def render_ai_diagnostics():
    if 'current_result' not in st.session_state:
        st.info("👈 Generate an ECG signal first to see AI diagnostics")
        return
    
    result = st.session_state.current_result
    
    st.subheader("🧠 Machine Learning Model Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Arrhythmia Classification")
        st.markdown(f"**Model Type:** {result['model_type']}")
        st.markdown(f"**Detected Condition:** {result['arrhythmia']['classification']}")
        st.markdown(f"**Confidence Score:** {result['arrhythmia']['confidence']*100:.2f}%")
        
        st.markdown("#### Class Probabilities")
        prob_df = pd.DataFrame({
            'Condition': list(result['arrhythmia']['probabilities'].keys()),
            'Probability': [v*100 for v in result['arrhythmia']['probabilities'].values()]
        })
        
        fig = go.Figure(go.Bar(
            x=prob_df['Probability'],
            y=prob_df['Condition'],
            orientation='h',
            marker_color='#1f77b4'
        ))
        fig.update_layout(
            xaxis_title="Probability (%)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key="arrhythmia_chart")
    
    with col2:
        st.markdown("### Emotional State Indicator (HRV-based)")
        st.caption("⚠ Estimated from HRV features. Not a direct measurement of true emotion.")
        st.markdown(f"**Estimated State:** {result['emotion']['emotion']}")
        st.markdown(f"**Valence:** {result['emotion']['valence']}")
        st.markdown(f"**Arousal Level:** {result['emotion']['arousal']}")
        
        st.markdown("#### Emotion Probabilities")

        if result['emotion']['emotion'] == "Unreliable":
          st.warning("⚠ Emotion probabilities hidden due to unreliable estimation under abnormal ECG conditions.")

        else:
             emo_df = pd.DataFrame({
             'Emotion': list(result['emotion']['probabilities'].keys()),
             'Probability': [v * 100 for v in result['emotion']['probabilities'].values()]
             })

             fig = go.Figure(go.Bar(
             x=emo_df['Probability'],
             y=emo_df['Emotion'],
             orientation='h',
             marker_color='#ff7f0e'
             ))

             st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.subheader("🔬 Feature Extraction Details")
    
    features = result['arrhythmia']['features']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean RR Interval", f"{features['mean_rr']:.2f} ms")
        st.metric("Std RR Interval", f"{features['std_rr']:.2f} ms")
    
    with col2:
        st.metric("QRS Duration", f"{features['qrs_duration']:.2f} ms")
        st.metric("HR Variability", f"{features['hr_variability']:.2f}")
    
    with col3:
        st.metric("Signal Energy", f"{features['signal_energy']:.4f}")
        st.metric("Peak Count", f"{features['peak_count']}")
    
    st.divider()

def render_historical_data():
    st.subheader("📋 Historical Analysis Records")
    
    if not st.session_state.history:
        st.info("No historical data yet. Generate some ECG analyses to build history.")
        return
    
    history_df = pd.DataFrame([
        {
            'Timestamp': h['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
            'Signal Type': h['signal_type'],
            'Heart Rate': f"{h['ecg_data']['heart_rate']:.0f} BPM",
            'Arrhythmia': h['arrhythmia']['classification'],
            'Confidence': f"{h['arrhythmia']['confidence']*100:.1f}%",
            'Stress Level': h['hrv']['stress_level'],
            'Emotional State': h['emotion']['emotion'],
            'Model': h['model_type'],
            'Device': h['edge_device'],
            'Inference Time': f"{h['edge_metrics']['inference_time_ms']:.2f} ms"
        }
        for h in st.session_state.history
    ])
    
    st.dataframe(history_df, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Heart Rate Trends")
        hr_data = [h['ecg_data']['heart_rate'] for h in st.session_state.history]
        timestamps = [h['timestamp'] for h in st.session_state.history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=hr_data,
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='#1f77b4', width=2)
        ))
        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title="Heart Rate (BPM)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True, key="emotion_chart")
    
    with col2:
        st.subheader("Arrhythmia Distribution")
        arrhythmia_counts = {}
        for h in st.session_state.history:
            arr_type = h['arrhythmia']['classification']
            arrhythmia_counts[arr_type] = arrhythmia_counts.get(arr_type, 0) + 1
        
        fig = go.Figure(go.Pie(
            labels=list(arrhythmia_counts.keys()),
            values=list(arrhythmia_counts.values()),
            hole=0.3
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export to CSV", use_container_width=True):
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"ecg_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📥 Export to JSON", use_container_width=True):
            json_data = json.dumps([
                {
                    'timestamp': h['timestamp'].isoformat(),
                    'signal_type': h['signal_type'],
                    'heart_rate': float(h['ecg_data']['heart_rate']),
                    'arrhythmia': h['arrhythmia']['classification'],
                    'stress_level': h['hrv']['stress_level'],
                    'emotional_state': h['emotion']['emotion']
                }
                for h in st.session_state.history
            ], indent=2)
            
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"ecg_analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def render_edge_metrics():
    if 'current_result' not in st.session_state:
        st.info("👈 Generate an ECG signal first to see Edge AI metrics")
        return
    
    result = st.session_state.current_result
    metrics = result['edge_metrics']
    
    st.subheader("⚡ Edge AI Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inference Time", f"{metrics['inference_time_ms']:.2f} ms")
    
    with col2:
        st.metric("Model Size", f"{metrics['model_size_kb']:.1f} KB")
    
    with col3:
        st.metric("Memory Usage", "Approx 200 KB")
        st.caption("Estimated runtime memory usage within device limits")
    
    with col4:
        st.metric("Power Consumption", f"{metrics['power_mw']:.1f} mW")
        st.caption("Estimated power usage during inference")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Target Device Specifications")
        st.markdown(f"**Device:** {result['edge_device']}")
        st.markdown(f"**Model:** {result['model_type']}")
        st.markdown(f"**CPU Clock:** {metrics['cpu_clock']} MHz")
        st.markdown(f"**RAM Available:** {metrics['ram_available']} KB")
        st.markdown(f"**Flash Storage:** {metrics['flash_storage']} KB")
        
        if metrics['inference_time_ms'] > 0:
            fps = 1000 / metrics['inference_time_ms']
        else:
            fps = 0
        st.markdown(f"**Processing Rate:** {fps:.2f} ECG segments/sec")
        st.caption("Number of ECG segments processed per second")
    
    with col2:
        st.subheader("📊 Resource Utilization")
        
        utilization_data = {
            'CPU': metrics['cpu_utilization'],
            'Memory': metrics['memory_utilization'],
            'Power': min(100, metrics['power_mw'] / 500 * 100)
        }
        
        fig = go.Figure()
        
        colors = ['#4caf50' if v < 70 else '#ff9800' if v < 90 else '#f44336' for v in utilization_data.values()]
        
        fig.add_trace(go.Bar(
            x=list(utilization_data.keys()),
            y=list(utilization_data.values()),
            marker_color=colors,
            text=[f"{v:.1f}%" for v in utilization_data.values()],
            textposition='auto'
        ))
        
        fig.update_layout(
            yaxis_title="Utilization (%)",
            yaxis_range=[0, 100],
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
