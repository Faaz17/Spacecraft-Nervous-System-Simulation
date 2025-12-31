# 🛰️ Spacecraft Nervous System Simulation
## Complete Project Documentation

---

## 📌 Project Overview

This project simulates a **Quantum-IoT Hybrid Network for Spacecraft Decision Systems** - essentially creating an intelligent "nervous system" for spacecraft that can:

1. **Sense** - Collect data from multiple sensor types
2. **Process** - Filter noise and fuse sensor data
3. **Decide** - Make autonomous decisions based on processed data
4. **Act** - Execute responses to maintain spacecraft health

**Course:** Quantum-IoT Hybrid Network for Spacecraft Decision Systems

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE SYSTEM PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌──────────────────┐    ┌──────────────────┐                      │
│   │ CLASSICAL SENSOR │    │  QUANTUM SENSOR  │                      │
│   │ (High noise,     │    │ (Low noise,      │                      │
│   │  Reliable)       │    │  Radiation-prone)│                      │
│   └────────┬─────────┘    └────────┬─────────┘                      │
│            │                       │                                 │
│            ▼                       ▼                                 │
│   ┌────────────────────────────────────────────┐                    │
│   │           LOW-PASS FILTER (DSP)            │                    │
│   │         Removes high-frequency noise        │                    │
│   └────────────────────┬───────────────────────┘                    │
│                        │                                             │
│                        ▼                                             │
│   ┌────────────────────────────────────────────┐                    │
│   │             SENSOR FUSION                   │                    │
│   │    Combines sensors, rejects anomalies     │                    │
│   └────────────────────┬───────────────────────┘                    │
│                        │                                             │
│            ┌───────────┴───────────┐                                │
│            ▼                       ▼                                 │
│   ┌─────────────────┐    ┌─────────────────┐                        │
│   │ RULE-BASED      │    │ AI-BASED        │                        │
│   │ DECISION ENGINE │    │ DECISION SYSTEM │                        │
│   │ (Thresholds)    │    │ (RL + Anomaly)  │                        │
│   └─────────────────┘    └─────────────────┘                        │
│            │                       │                                 │
│            └───────────┬───────────┘                                │
│                        ▼                                             │
│   ┌────────────────────────────────────────────┐                    │
│   │           SPACECRAFT ACTUATORS              │                    │
│   │    (Shield, Thrusters, Safe Mode)          │                    │
│   └────────────────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project File Structure

| File | Purpose |
|------|---------|
| `data_gen.py` | Generates synthetic telemetry data (classical & quantum sensors) |
| `data_loader.py` | Loads real sensor data from hardware team's CSV files |
| `dsp.py` | Digital Signal Processing (Butterworth Low-Pass Filter) |
| `fusion.py` | Sensor fusion with anomaly detection/rejection |
| `decision.py` | Rule-based decision engine (NOMINAL/WARNING/CRITICAL) |
| `anomaly.py` | PyTorch Autoencoder for AI-based anomaly detection |
| `rl_env.py` | Gymnasium environment for Reinforcement Learning |
| `main.py` | Basic simulation with mock data |
| `main_real_data.py` | Simulation using real hardware data |
| `main_ai.py` | Full AI system (Anomaly Detection + RL Agent) |
| `main_final_integration.py` | Deploys trained AI on real hardware data |
| `app.py` | Streamlit interactive dashboard |
| `requirements.txt` | Python dependencies |

### Data Folders

| Folder | Contents |
|--------|----------|
| `Quantum IoT Network Simulation Vanshika/` | Vanshika's hardware simulation (Steps 1-2) |
| `Quantum IoT Network Simulation Varenya/` | Varenya's hardware simulation (Steps 1-4) |
| `ai_models/` | Trained AI models (anomaly_detector.pt, ppo_spacecraft.zip) |

---

## 🔬 Module Details

### 1. Data Generation (`data_gen.py`)

**Purpose:** Simulates realistic spacecraft sensor readings

**Physical Model:**
- **Ground Truth:** 1 Hz sinusoidal vibration (structural oscillation)
- **Classical Sensor:** Ground truth + Gaussian noise (σ=0.5) + thermal drift
- **Quantum Sensor:** Ground truth + low noise (σ=0.05) + radiation spikes (~1% of samples)

```python
from data_gen import generate_telemetry, get_spike_indices

time_axis, ground_truth, classical_output, quantum_output = generate_telemetry(
    duration=10.0,   # seconds
    rate=100,        # Hz
    seed=42          # reproducibility
)

spike_indices = get_spike_indices(quantum_output, threshold=3.0)
```

**Key Insight:** Classical sensors are noisy but reliable; Quantum sensors are precise but fail catastrophically when hit by radiation.

---

### 2. Data Loader (`data_loader.py`)

**Purpose:** Loads real sensor data from hardware simulation team

**Data Sources:**

| Source | Stage | Description |
|--------|-------|-------------|
| Vanshika | Steps 1-2 | Raw sensor outputs |
| Varenya | `raw` | Steps 1-2 basic output |
| Varenya | `env` | Step 3 with environmental disturbances |
| Varenya | `adc` | Step 4 after ADC conversion |

```python
from data_loader import load_vanshika_data, load_varenya_data, SensorData

# Load Vanshika's data
data = load_vanshika_data()

# Load Varenya's data (different stages)
data = load_varenya_data(use_stage='raw')  # Steps 1-2
data = load_varenya_data(use_stage='env')  # Step 3
data = load_varenya_data(use_stage='adc')  # Step 4

# SensorData contains:
# - time: np.ndarray
# - ground_truth: np.ndarray
# - classical: np.ndarray
# - quantum: np.ndarray
# - sample_rate: float
# - duration: float
# - source: str
```

---

### 3. Digital Signal Processing (`dsp.py`)

**Purpose:** Removes high-frequency noise while preserving the signal of interest

**Implementation:**
- **Butterworth Low-Pass Filter** (maximally flat passband)
- **Zero-phase filtering** using `filtfilt` (no time delay)
- Default cutoff: 5 Hz for mock data, 50 Hz for real data

```python
from dsp import LowPassFilter

lpf = LowPassFilter(
    cutoff_freq=5.0,    # Hz - frequencies above this are attenuated
    sample_rate=100,    # Hz - must match data sampling rate
    order=4             # filter order (higher = sharper cutoff)
)

filtered_signal = lpf.apply(noisy_signal)

# Get frequency response for analysis
frequencies, magnitude_db = lpf.get_frequency_response()
```

**Why Low-Pass?** Ground truth is ~1 Hz; higher frequencies are noise.

**Filter Theory:**
- Butterworth has maximally flat frequency response in passband
- `filtfilt` applies filter forward and backward for zero phase distortion
- Critical for sensor fusion where time alignment matters

---

### 3.1 Signal-to-Noise Ratio (SNR) Analysis

**Purpose:** Quantify the effectiveness of noise filtering

**SNR Formula:**
```
SNR (dB) = 10 × log₁₀(Signal Power / Noise Power)

Where:
- Signal Power = Variance of Ground Truth
- Noise Power = Variance of (Measured Signal - Ground Truth)
```

**How Filtering Improves SNR:**

| Stage | Classical SNR | Quantum SNR |
|-------|--------------|-------------|
| Raw (before filter) | ~3-6 dB | ~20-25 dB |
| After Low-Pass Filter | ~10-15 dB | ~25-30 dB |
| Fused Signal | ~15-25 dB | - |

**Noise Reduction Calculation:**
```python
noise_reduction_pct = 100 × (1 - filtered_noise_power / raw_noise_power)
```

**Why Low-Pass Filtering Works:**
1. **Ground truth signal is low frequency** (~1-11 Hz structural vibration)
2. **Noise is broadband** (spreads across all frequencies)
3. **Cutting high frequencies** removes noise without losing signal
4. **Result:** Higher SNR = cleaner signal for decision making

**Typical Results:**
- Classical sensor noise reduction: 40-60%
- Quantum sensor noise reduction: 30-50%
- Final fused signal correlation with ground truth: >0.99

---

### 4. Sensor Fusion (`fusion.py`)

**Purpose:** Intelligently combines classical and quantum sensors

**Algorithm:**
```
For each time step t:
    disagreement = |Classical(t) - Quantum(t)|
    
    If disagreement > threshold:
        # Anomaly detected - quantum hit by radiation
        Fused(t) = Classical(t)  # Use reliable classical only
    Else:
        # Normal operation - weighted average
        Fused(t) = 0.8 × Quantum(t) + 0.2 × Classical(t)
```

**Rationale:**
- Quantum is 10× more precise → favor it when working
- Classical is always reliable → use as backup
- 80/20 weighting balances precision with robustness

```python
from fusion import fuse_signals, calculate_fusion_quality, FusionResult

result = fuse_signals(
    classical=classical_filtered,
    quantum=quantum_filtered,
    threshold=2.0,          # max allowed disagreement
    quantum_weight=0.8      # 80% quantum, 20% classical
)

# FusionResult contains:
# - fused_signal: np.ndarray - the combined output
# - anomaly_indices: List[int] - where quantum was rejected
# - trust_quantum: np.ndarray - boolean mask of trust

# Evaluate quality
metrics = calculate_fusion_quality(result.fused_signal, ground_truth)
# Returns: rmse, mae, max_error, correlation
```

---

### 5. Decision Engine (`decision.py`)

**Purpose:** Translates sensor data into spacecraft actions using a Finite State Machine

**State Diagram:**
```
┌──────────┐        ┌──────────┐        ┌──────────┐
│ NOMINAL  │ ──────>│ WARNING  │ ──────>│ CRITICAL │
│ (Green)  │<────── │ (Yellow) │<────── │  (Red)   │
└──────────┘        └──────────┘        └──────────┘

|signal| < 0.8      0.8 ≤ |signal| ≤ 1.5    |signal| > 1.5
```

| State | Condition | Action |
|-------|-----------|--------|
| **NOMINAL** | \|signal\| < 0.8 | Continue Data Logging - All Systems Normal |
| **WARNING** | 0.8 ≤ \|signal\| ≤ 1.5 | Fire Attitude Thrusters - Dampen Oscillation |
| **CRITICAL** | \|signal\| > 1.5 | EMERGENCY: Initiate Safe Mode - Shutdown Non-Essential Systems |

```python
from decision import DecisionEngine, SystemState, Decision

engine = DecisionEngine(
    nominal_threshold=0.8,
    warning_threshold=1.5,
    processing_latency_ms=5.0,
    simulate_latency=False
)

# Single evaluation
decision = engine.evaluate(signal_value=0.5, timestamp=1.0)
# decision.state: SystemState.NOMINAL
# decision.action: "Continue Data Logging..."

# Batch evaluation
decisions = engine.evaluate_signal_array(fused_signal, time_axis)

# Get statistics
engine.print_summary()
timestamps, state_values = engine.get_state_timeline()
```

---

### 6. Anomaly Detection (`anomaly.py`)

**Purpose:** AI-based anomaly detection using PyTorch Autoencoder

**Architecture:**
```
Input (window) → Encoder → Latent Space → Decoder → Reconstructed
     (15)        (16→8→6)     (6)        (6→8→16)      (15)
```

**Training Philosophy:**
1. Train ONLY on normal data (no anomalies!)
2. Model learns to reconstruct normal patterns
3. During inference: Anomalous data → High reconstruction error → Detected!

```python
from anomaly import AnomalyDetector, AnomalyResult

detector = AnomalyDetector(
    window_size=15,              # consecutive samples per window
    latent_dim=6,                # bottleneck size
    threshold_percentile=99.5,   # for anomaly threshold
    device='cpu'                 # or 'cuda' for GPU
)

# Training (only on NORMAL data!)
history = detector.train(
    normal_data,
    epochs=80,
    batch_size=32,
    learning_rate=0.001,
    verbose=True
)

# Single detection
result = detector.detect(data_window)
# result.is_anomaly: bool
# result.reconstruction_error: float
# result.threshold: float
# result.confidence: float

# Batch detection
anomaly_flags, reconstruction_errors = detector.detect_batch(full_signal)

# Save/Load model
detector.save("ai_models/anomaly_detector.pt")
detector.load("ai_models/anomaly_detector.pt")
```

**Applications:**
- Radiation spike detection
- Structural stress anomalies
- Communication delay detection
- Thermal anomaly detection

---

### 7. Reinforcement Learning Environment (`rl_env.py`)

**Purpose:** Gymnasium-compatible environment for training autonomous agents

**State Space (Observation):**
```python
[signal_value, health, energy, reconstruction_error]
# Ranges: [-5, 5], [0, 100], [0, 100], [0, 1]
```

**Action Space:**

| Action | Name | Energy Effect | During Anomaly | During Safe |
|--------|------|--------------|----------------|-------------|
| 0 | Do Nothing | +0.5 (recharge) | -6 reward, -5 health | +3 reward |
| 1 | Shield | -0.1 | +1.5 reward | -7 reward (wasteful) |
| 2 | Thruster | -1.0 | +0.5 reward | -8 reward (wasteful) |

**Goal:** Agent learns to shield only when necessary, conserving energy while surviving.

```python
from rl_env import SpacecraftEnv
import gymnasium as gym
from stable_baselines3 import PPO

# Create environment
env = SpacecraftEnv(max_steps=1000, initial_health=100.0, initial_energy=100.0)

# Set real data
env.set_data(fused_signal, anomaly_flags, reconstruction_errors, threshold)

# Train PPO agent
model = PPO("MlpPolicy", env, learning_rate=0.0003, n_steps=512, verbose=1)
model.learn(total_timesteps=30000, progress_bar=True)

# Save/Load
model.save("ai_models/ppo_spacecraft")
model = PPO.load("ai_models/ppo_spacecraft")

# Deployment
obs, info = env.reset()
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

stats = env.get_episode_stats()
```

---

## 🚀 Execution Workflows

### Workflow 1: Basic Simulation (`main.py`)

```bash
python main.py
```

**Steps:**
1. Generate 10 seconds of mock telemetry (100 Hz)
2. Apply low-pass filter (5 Hz cutoff)
3. Fuse classical + quantum with anomaly rejection
4. Run decision engine on fused signal
5. Generate visualization dashboard

**Output:** `simulation_dashboard.png`

---

### Workflow 2: Real Data Simulation (`main_real_data.py`)

```bash
python main_real_data.py                    # Default: Varenya raw
python main_real_data.py --source vanshika  # Vanshika's data
python main_real_data.py --stage adc        # Varenya ADC stage
python main_real_data.py --cutoff 100       # Custom filter cutoff
```

**Arguments:**
- `--source`: `vanshika` or `varenya` (default: varenya)
- `--stage`: `raw`, `env`, or `adc` (default: raw)
- `--cutoff`: Filter cutoff frequency in Hz (default: 50.0)

**Output:** `dashboard_varenya_{stage}.png` or `dashboard_vanshika.png`

---

### Workflow 3: AI Decision System (`main_ai.py`)

```bash
python main_ai.py
```

**Three-Phase Execution:**

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ANOMALY DETECTOR TRAINING                                    │
│  • Generate 2000 normal samples (sine + harmonics)                     │
│  • Train autoencoder for 80 epochs                                     │
│  • Set threshold at 99.5th percentile of reconstruction error          │
│  → Output: ai_models/anomaly_detector.pt                               │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: RL AGENT TRAINING                                            │
│  • Create SpacecraftEnv with sensor + anomaly data                     │
│  • Train PPO agent for 30,000 timesteps                                │
│  • Agent learns: when to shield, when to idle, conserve energy         │
│  → Output: ai_models/ppo_spacecraft.zip                                │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: DEPLOYMENT SIMULATION                                        │
│  • Generate test data with real anomalies                              │
│  • Fuse sensors, detect anomalies with autoencoder                     │
│  • RL agent makes real-time decisions                                  │
│  → Output: ai_decision_system_{timestamp}.png                          │
└────────────────────────────────────────────────────────────────────────┘
```

**Configuration (in Config class):**
```python
NORMAL_DATA_DURATION = 20      # seconds of training data
ANOMALY_WINDOW_SIZE = 15       # samples per window
ANOMALY_EPOCHS = 80            # training epochs
RL_TIMESTEPS = 30000           # PPO training steps
```

---

### Workflow 4: Final Integration (`main_final_integration.py`)

```bash
python main_final_integration.py
```

**Purpose:** Deploy trained AI models on real hardware data

**Requirements:** Must run `main_ai.py` first to generate trained models!

**Steps:**
1. Load real sensor data (Varenya's ADC output)
2. Apply DSP filtering and sensor fusion
3. Load trained Anomaly Detector (ai_models/anomaly_detector.pt)
4. Load trained PPO Agent (ai_models/ppo_spacecraft.zip)
5. Run frame-by-frame decision loop on real data
6. Visualize AI decisions

**Output:** `FINAL_INTEGRATED_DEMO.png`

---

### Workflow 5: Interactive Dashboard (`app.py`)

```bash
streamlit run app.py
```

**Features:**
- Web-based interactive UI at http://localhost:8501
- Switch between mock and real data sources
- Adjustable parameters (duration, filter cutoff, seed)
- Real-time visualization with Plotly
- Compare results across different data stages
- Pipeline diagram visualization

**Data Sources Available:**
- 🎲 Mock Data (Generated)
- 📡 Varenya - Steps 1-2: Raw Sensors
- 🌡️ Varenya - Step 3: Environmental Disturbances
- 📟 Varenya - Step 4: ADC Output

---

## 📊 Key Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **RMSE** | Root Mean Square Error vs Ground Truth | < 0.1 |
| **MAE** | Mean Absolute Error | < 0.1 |
| **Correlation** | Pearson correlation with Ground Truth | > 0.99 |
| **Anomalies Detected** | Radiation spikes found | ~1% of samples |
| **Anomalies Rejected** | Quantum readings discarded | ≈ Detected |
| **Final Health** (AI) | Spacecraft health after simulation | > 80% |
| **Total Reward** (AI) | Cumulative RL reward | > 0 |

---

## 🧠 AI System Details

### Why AI?
- Rule-based systems are brittle (fixed thresholds fail in edge cases)
- AI learns patterns in data automatically
- RL agent balances multiple objectives (health vs energy vs action cost)
- Generalizes to unseen scenarios

### Autoencoder Anomaly Detection

**Key Insight:** "Train on normal, detect abnormal"

1. Autoencoder compresses and reconstructs data
2. When trained only on normal patterns, it learns "what normal looks like"
3. Anomalous inputs cannot be reconstructed well → High error
4. High reconstruction error = Anomaly detected!

**Advantages:**
- No labeled anomaly data needed
- Generalizes to new types of anomalies
- Computationally efficient at inference

### PPO Reinforcement Learning

**Algorithm:** Proximal Policy Optimization (PPO)
- State-of-the-art policy gradient method
- Stable training with clipped objective
- Balances exploration and exploitation

**What the Agent Learns:**
1. Recognize when action is needed (high reconstruction error)
2. Choose minimal sufficient response (shield vs thruster vs idle)
3. Conserve energy for when it's truly needed
4. Survive as long as possible with maximum health

**Reward Shaping:**
```python
# During anomaly (must respond!)
if action == Shield:  reward += 1.5
if action == Idle:    reward -= 6.0, health -= 5

# During safe period (should conserve)
if action == Shield:  reward -= 7.0  # wasteful!
if action == Idle:    reward += 3.0  # good!
```

---

## 📦 Installation

### Requirements
```
numpy>=1.21.0          # Numerical computing
scipy>=1.7.0           # Signal processing
pandas>=1.3.0          # Data loading
matplotlib>=3.5.0      # Static plots
streamlit>=1.28.0      # Web dashboard
plotly>=5.18.0         # Interactive plots
torch>=2.0.0           # Deep learning
gymnasium>=0.29.0      # RL environment
stable-baselines3>=2.1.0  # RL algorithms
tqdm>=4.65.0           # Progress bars
rich>=13.0.0           # Rich output
```

### Installation Steps

```bash
# 1. Clone/download the project

# 2. Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run basic simulation
python main.py

# 5. Run AI system
python main_ai.py

# 6. Run interactive dashboard
streamlit run app.py
```

---

## 🎯 Key Findings

1. **Quantum sensors provide 10× precision improvement** when working correctly (σ=0.05 vs σ=0.5)

2. **Fusion algorithm successfully rejects radiation spikes** using sensor disagreement detection

3. **Final fused signal tracks ground truth with >99% correlation**

4. **AI system learns intelligent behavior:**
   - Shields only during actual anomalies
   - Idles during safe periods to conserve energy
   - Maintains high health throughout mission

5. **Rule-based and AI approaches are complementary:**
   - Rules provide safety guarantees and interpretability
   - AI provides optimization and adaptation

---

## 👥 Team

- **Hardware Simulation:** Vanshika & Varenya (MATLAB)
- **Software Stack:** Signal Processing & Decision Engine (Python)

---

## 📁 Output Files

| File | Source | Description |
|------|--------|-------------|
| `simulation_dashboard.png` | main.py | Basic mock data visualization |
| `dashboard_varenya_*.png` | main_real_data.py | Real data visualization |
| `ai_decision_system_*.png` | main_ai.py | AI system performance |
| `FINAL_INTEGRATED_DEMO.png` | main_final_integration.py | AI on real data |
| `ai_models/anomaly_detector.pt` | main_ai.py | Trained autoencoder |
| `ai_models/ppo_spacecraft.zip` | main_ai.py | Trained RL agent |

---

## 🔗 Quick Reference

```bash
# Basic simulation (mock data)
python main.py

# Real data simulation
python main_real_data.py --stage adc

# Train AI system
python main_ai.py

# Deploy AI on real data
python main_final_integration.py

# Interactive dashboard
streamlit run app.py
```

---

*Generated: December 29, 2025*
*Spacecraft Nervous System Simulation Team*

