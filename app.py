"""
QUANTUM-IoT SPACECRAFT NERVOUS SYSTEM - INTERACTIVE DASHBOARD
RUN: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_gen import generate_telemetry, get_spike_indices
from data_loader import load_varenya_data, detect_radiation_spikes
from dsp import LowPassFilter
from fusion import fuse_signals, calculate_fusion_quality
from decision import DecisionEngine, SystemState

st.set_page_config(page_title="Spacecraft Nervous System", page_icon="🛰️", layout="wide")

# =============================================================================
# SIMULATION FUNCTIONS
# =============================================================================

def run_mock_sim(duration, sample_rate, filter_cutoff, seed):
    time_axis, ground_truth, classical, quantum = generate_telemetry(duration=duration, rate=sample_rate, seed=seed)
    spike_indices = get_spike_indices(quantum, threshold=3.0)
    lpf = LowPassFilter(cutoff_freq=filter_cutoff, sample_rate=sample_rate, order=4)
    classical_filtered, quantum_filtered = lpf.apply(classical), lpf.apply(quantum)
    fusion_result = fuse_signals(classical_filtered, quantum_filtered, threshold=2.0, quantum_weight=0.8)
    quality = calculate_fusion_quality(fusion_result.fused_signal, ground_truth)
    engine = DecisionEngine(nominal_threshold=0.8, warning_threshold=1.5, simulate_latency=False)
    decisions = engine.evaluate_signal_array(fusion_result.fused_signal, time_axis)
    return {'time': time_axis, 'ground_truth': ground_truth, 'classical_raw': classical, 
            'quantum_raw': quantum, 'fused': fusion_result.fused_signal, 'spike_indices': spike_indices,
            'anomaly_indices': fusion_result.anomaly_indices, 'quality': quality, 'decisions': decisions, 
            'engine': engine, 'source': 'Mock Data'}

def run_real_sim(data, filter_cutoff, source_name):
    lpf = LowPassFilter(cutoff_freq=filter_cutoff, sample_rate=data.sample_rate, order=4)
    classical_filtered, quantum_filtered = lpf.apply(data.classical), lpf.apply(data.quantum)
    threshold = 2.0 * np.std(data.classical)
    fusion_result = fuse_signals(classical_filtered, quantum_filtered, threshold=threshold, quantum_weight=0.8)
    quality = calculate_fusion_quality(fusion_result.fused_signal, data.ground_truth)
    spike_indices = detect_radiation_spikes(data.quantum, data.ground_truth)
    amp = np.percentile(np.abs(data.ground_truth), 95)
    engine = DecisionEngine(nominal_threshold=0.8*amp, warning_threshold=1.2*amp, simulate_latency=False)
    decisions = engine.evaluate_signal_array(fusion_result.fused_signal, data.time)
    return {'time': data.time, 'ground_truth': data.ground_truth, 'classical_raw': data.classical,
            'quantum_raw': data.quantum, 'fused': fusion_result.fused_signal, 'spike_indices': spike_indices,
            'anomaly_indices': fusion_result.anomaly_indices, 'quality': quality, 'decisions': decisions, 
            'engine': engine, 'source': source_name}

# =============================================================================
# PLOTTING
# =============================================================================

def create_sensor_plot(results):
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Classical Sensor (Noisy, Reliable)', 
        'Quantum Sensor (Precise, Radiation-Vulnerable)', 'Ground Truth'), vertical_spacing=0.08)
    t = results['time']
    fig.add_trace(go.Scatter(x=t, y=results['classical_raw'], name='Classical', line=dict(color='#ff6b6b', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['ground_truth'], name='Truth', line=dict(color='#ffe66d', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['quantum_raw'], name='Quantum', line=dict(color='#4ecdc4', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['ground_truth'], name='Truth', line=dict(color='#ffe66d', width=2, dash='dash'), showlegend=False), row=2, col=1)
    if len(results['spike_indices']) > 0:
        fig.add_trace(go.Scatter(x=t[results['spike_indices']], y=results['quantum_raw'][results['spike_indices']],
            mode='markers', name=f'Spikes ({len(results["spike_indices"])})', marker=dict(color='#ff4757', size=8, symbol='x')), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['ground_truth'], name='Ground Truth', line=dict(color='#ffe66d', width=2)), row=3, col=1)
    fig.update_layout(height=550, template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e')
    return fig

def create_fusion_plot(results):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Sensor Fusion Result', 'Error from Ground Truth'), vertical_spacing=0.12)
    t = results['time']
    fig.add_trace(go.Scatter(x=t, y=results['ground_truth'], name='Ground Truth', line=dict(color='#ffe66d', width=2, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=results['fused'], name='Fused Signal', line=dict(color='#95e1d3', width=1.5)), row=1, col=1)
    if len(results['anomaly_indices']) > 0:
        fig.add_trace(go.Scatter(x=t[results['anomaly_indices']], y=results['fused'][results['anomaly_indices']],
            mode='markers', name=f'Anomalies Rejected ({len(results["anomaly_indices"])})', marker=dict(color='#ffa502', size=6)), row=1, col=1)
    error = results['fused'] - results['ground_truth']
    fig.add_trace(go.Scatter(x=t, y=error, name='Error', fill='tozeroy', fillcolor='rgba(149,225,211,0.3)', line=dict(color='#95e1d3', width=1)), row=2, col=1)
    fig.update_layout(height=400, template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e')
    return fig

def create_state_plot(results):
    state_map = {SystemState.NOMINAL: 0, SystemState.WARNING: 1, SystemState.CRITICAL: 2}
    colors = {0: '#2ed573', 1: '#ffa502', 2: '#ff4757'}
    names = {0: 'NOMINAL', 1: 'WARNING', 2: 'CRITICAL'}
    states = [state_map[d.state] for d in results['decisions']]
    t = results['time']
    fig = go.Figure()
    for i in range(len(t) - 1):
        fig.add_shape(type="rect", x0=t[i], x1=t[i+1], y0=0, y1=1, fillcolor=colors[states[i]], opacity=0.7, line_width=0)
    for s, c in colors.items():
        pct = 100 * states.count(s) / len(states)
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(size=15, color=c, symbol='square'), name=f'{names[s]}: {pct:.1f}%'))
    fig.update_layout(height=150, template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e',
        xaxis_title='Time (s)', yaxis=dict(visible=False), legend=dict(orientation="h", y=1.3), margin=dict(t=30, b=30))
    fig.update_xaxes(range=[t[0], t[-1]])
    return fig

def show_results(results):
    st.subheader(f"📊 Results: {results['source']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Samples", len(results['time']))
    c2.metric("Spikes Found", len(results['spike_indices']))
    c3.metric("Anomalies Rejected", len(results['anomaly_indices']))
    c4.metric("RMSE", f"{results['quality']['rmse']:.4f}")
    c5.metric("Correlation", f"{results['quality']['correlation']:.4f}")
    st.plotly_chart(create_sensor_plot(results), use_container_width=True)
    st.plotly_chart(create_fusion_plot(results), use_container_width=True)
    st.markdown("### Decision Timeline")
    st.plotly_chart(create_state_plot(results), use_container_width=True)

def show_comparison():
    """Show comparison of all saved results"""
    comparison = []
    
    if 'mock_results' in st.session_state:
        r = st.session_state['mock_results']
        comparison.append({
            'Source': 'Mock Data',
            'Samples': len(r['time']),
            'Spikes': len(r['spike_indices']),
            'Rejected': len(r['anomaly_indices']),
            'RMSE': round(r['quality']['rmse'], 4),
            'Correlation': round(r['quality']['correlation'], 4)
        })
    
    for stage, name in [('raw', 'Steps 1-2'), ('env', 'Step 3'), ('adc', 'Step 4')]:
        key = f'varenya_{stage}_results'
        if key in st.session_state:
            r = st.session_state[key]
            comparison.append({
                'Source': f'Varenya {name}',
                'Samples': len(r['time']),
                'Spikes': len(r['spike_indices']),
                'Rejected': len(r['anomaly_indices']),
                'RMSE': round(r['quality']['rmse'], 4),
                'Correlation': round(r['quality']['correlation'], 4)
            })
    
    if len(comparison) > 0:
        st.markdown("---")
        st.subheader("📈 Compare All Results")
        
        df = pd.DataFrame(comparison)
        st.dataframe(df, use_container_width=True)
        
        if len(comparison) > 1:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(df, x='Source', y='RMSE', title='RMSE (Lower = Better)', 
                    color='RMSE', color_continuous_scale='Reds')
                fig.update_layout(template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e', showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(df, x='Source', y='Correlation', title='Correlation (Higher = Better)',
                    color='Correlation', color_continuous_scale='Greens')
                fig.update_layout(template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#16213e', showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

def create_pipeline():
    fig = go.Figure()
    blocks = [('CLASSICAL\nSENSOR', 0, 1, '#ff6b6b'), ('QUANTUM\nSENSOR', 0, 0, '#4ecdc4'),
              ('LOW-PASS\nFILTER', 1, 0.5, '#9b59b6'), ('SENSOR\nFUSION', 2, 0.5, '#3498db'),
              ('DECISION\nENGINE', 3, 0.5, '#2ed573'), ('ACTUATOR', 4, 0.5, '#f39c12')]
    for name, x, y, color in blocks:
        fig.add_shape(type="rect", x0=x-0.35, x1=x+0.35, y0=y-0.25, y1=y+0.25, fillcolor=color, opacity=0.8, line=dict(color='white', width=2))
        fig.add_annotation(x=x, y=y, text=name, showarrow=False, font=dict(color='white', size=9))
    for x0, y0, x1, y1 in [(0,1,1,0.5), (0,0,1,0.5), (1,0.5,2,0.5), (2,0.5,3,0.5), (3,0.5,4,0.5)]:
        fig.add_annotation(x=x1-0.4, y=y1, ax=x0+0.4, ay=y0, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor='#eee')
    fig.update_layout(height=180, template='plotly_dark', paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e',
        xaxis=dict(visible=False, range=[-0.5, 4.5]), yaxis=dict(visible=False, range=[-0.5, 1.5]), margin=dict(l=10, r=10, t=10, b=10))
    return fig

# =============================================================================
# MAIN APP - ALL IN ONE PAGE
# =============================================================================

def main():
    st.markdown('<h1 style="text-align:center;color:#4ecdc4;">🛰️ Spacecraft Nervous System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#888;">Quantum-IoT Hybrid Network for Decision Systems</p>', unsafe_allow_html=True)
    st.plotly_chart(create_pipeline(), use_container_width=True)
    
    tab1, tab2 = st.tabs(["🔬 Simulation & Results", "📚 About"])
    
    # === TAB 1: EVERYTHING IN ONE ===
    with tab1:
        st.header("Run Simulation")
        
        # Data source selection
        data_source = st.selectbox("📊 Select Data Source", [
            "🎲 Mock Data (Generated)",
            "📡 Varenya - Steps 1-2: Raw Sensors",
            "🌡️ Varenya - Step 3: Environmental Disturbances", 
            "📟 Varenya - Step 4: ADC Output"
        ])
        
        st.markdown("---")
        
        # Different parameters based on selection
        if "Mock" in data_source:
            st.markdown("**Mock Data Parameters**")
            c1, c2, c3, c4 = st.columns(4)
            duration = c1.slider("Duration (s)", 5, 30, 10)
            sample_rate = c2.selectbox("Sample Rate (Hz)", [50, 100, 200], index=1)
            filter_cutoff = c3.slider("Filter Cutoff (Hz)", 1, 20, 5)
            seed = c4.number_input("Random Seed", 1, 9999, 42)
            
            if st.button("🚀 Run Mock Simulation", type="primary", use_container_width=True):
                with st.spinner("Running simulation..."):
                    results = run_mock_sim(duration, sample_rate, filter_cutoff, seed)
                    st.session_state['current_results'] = results
                    st.session_state['mock_results'] = results
                    
        else:
            # Varenya's data
            if "Steps 1-2" in data_source:
                stage = 'raw'
                stage_name = "Varenya Steps 1-2 (Raw)"
            elif "Step 3" in data_source:
                stage = 'env'
                stage_name = "Varenya Step 3 (Environmental)"
            else:
                stage = 'adc'
                stage_name = "Varenya Step 4 (ADC)"
            
            try:
                data = load_varenya_data(use_stage=stage)
                st.success(f"✅ Loaded: {len(data.time)} samples, {data.duration:.2f}s @ {data.sample_rate:.0f}Hz")
                
                filter_cutoff = st.slider("Filter Cutoff (Hz)", 10, 100, 50)
                
                if st.button(f"🔬 Analyze {stage_name}", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        results = run_real_sim(data, filter_cutoff, stage_name)
                        st.session_state['current_results'] = results
                        st.session_state[f'varenya_{stage}_results'] = results
                        
            except Exception as e:
                st.error(f"Error loading data: {e}")
        
        # Show current results
        st.markdown("---")
        if 'current_results' in st.session_state:
            show_results(st.session_state['current_results'])
        
        # Show comparison if we have multiple results
        show_comparison()
    
    # === TAB 2: ABOUT ===
    with tab2:
        st.header("About This Project")
        st.markdown("""
        ## Quantum-IoT Hybrid Network for Spacecraft Decision Systems
        
        ### Problem
        Spacecraft sensors face extreme conditions:
        - **Radiation** causes random spikes in quantum sensors
        - **Thermal noise** affects classical sensors
        - **Mission-critical decisions** depend on accurate data
        
        ### Solution: Hybrid Sensor Fusion
        
        | Sensor Type | Noise Level | Precision | Failure Mode |
        |-------------|-------------|-----------|--------------|
        | **Classical** | High (σ=0.5) | Low | Gradual drift |
        | **Quantum** | Low (σ=0.05) | High | Radiation spikes |
        
        Our fusion algorithm:
        1. **Detects** when quantum sensor has radiation hit
        2. **Rejects** anomalous quantum readings
        3. **Combines** both sensors optimally (80% quantum, 20% classical)
        
        ### Data Sources
        
        - **Mock Data**: Synthetic data for testing the pipeline
        - **Varenya Steps 1-2**: Raw sensor outputs from hardware simulation
        - **Varenya Step 3**: With environmental disturbances (temperature, EMI)
        - **Varenya Step 4**: After ADC (Analog-to-Digital Conversion)
        
        ### System Pipeline
        ```
        Classical Sensor ─┐
                          ├─→ Low-Pass Filter → Sensor Fusion → Decision Engine → Actuator
        Quantum Sensor ───┘
        ```
        
        ### Team
        - **Hardware Simulation**: Vanshika & Varenya (MATLAB)
        - **Software Stack**: Signal Processing & Decision Engine (Python)
        """)

if __name__ == "__main__":
    main()
