%% ========================================================================
%% STEP 0: SENSOR SIMULATION (Defining 't' and signals)
%% ========================================================================
clear; clc; close all;

% Parameters
Fs = 1000;              % Sampling frequency (Hz)
T = 5;                  % Duration (seconds)
t = 0:1/Fs:T-1/Fs;      % Time vector

% 1. Create Ground Truth Signal (Spacecraft vibration/signal)
signal_perfect = sin(2*pi*5*t) + 0.5*sin(2*pi*12*t); 

% 2. Classical Sensor Noise Modeling
classical_thermal   = 0.05 * randn(size(t));
classical_flicker   = 0.02 * cumsum(randn(size(t))) / 100;
classical_drift     = 0.005 * t;
classical_radiation = zeros(size(t)); 
classical_radiation(mod(t, 1.2) < 0.02) = 1.5; % Periodic spikes

classical_noise_total = classical_thermal + classical_flicker + classical_drift + classical_radiation;
classical_sensor_output = signal_perfect + classical_noise_total;

% 3. Quantum Sensor Noise Modeling
quantum_phase_noise    = 0.03 * randn(size(t));
quantum_decoherence    = 0.1 * exp(-t/2) .* randn(size(t));
environmental_B_noise  = 0.04 * sin(2*pi*50*t); % Power line interference
quantum_init_errors    = zeros(size(t));
quantum_init_errors(rand(size(t)) > 0.995) = 0.8; % Random bit flips

quantum_noise_total = quantum_phase_noise + quantum_decoherence + environmental_B_noise + quantum_init_errors;
quantum_sensor_output = signal_perfect + quantum_noise_total;

% 4. Calculate Statistics
SNR_classical_sensor = 10 * log10(var(signal_perfect) / var(classical_noise_total));
SNR_quantum_sensor   = 10 * log10(var(signal_perfect) / var(quantum_noise_total));

rms_classical = sqrt(mean((classical_sensor_output - signal_perfect).^2));
rms_quantum   = sqrt(mean((quantum_sensor_output - signal_perfect).^2));


%% ========================================================================
%% STEP 3: ENVIRONMENTAL DISTURBANCE INJECTION
%% ========================================================================
fprintf('Step 3: Injecting environmental disturbances...\n')

% 3.1 Radiation-Induced Transient Faults
% Classical sensor: Occasional spikes/dropouts
radiation_fault_classical = zeros(size(t));
fault_indices_classical = find(rand(size(t)) > 0.98);  % ~2% of samples affected
radiation_fault_classical(fault_indices_classical) = 2.5 * randn(length(fault_indices_classical), 1);

% Quantum sensor: More sensitive to radiation transients
radiation_fault_quantum = zeros(size(t));
fault_indices_quantum = find(rand(size(t)) > 0.96);  % ~4% of samples affected
radiation_fault_quantum(fault_indices_quantum) = 3.0 * randn(length(fault_indices_quantum), 1);

% 3.2 Thermal Drift and Jitter
% Thermal drift: Low-frequency drift due to temperature variations
thermal_drift_rate_classical = 0.008;  % Slightly higher drift than original
thermal_drift_classical = thermal_drift_rate_classical * t .* (1 + 0.1*sin(2*pi*0.1*t));

thermal_drift_rate_quantum = 0.012;    % Quantum sensors more susceptible to thermal effects
thermal_drift_quantum = thermal_drift_rate_quantum * t .* (1 + 0.15*sin(2*pi*0.08*t));

% Jitter: High-frequency timing uncertainty (~0.1% of sampling period)
jitter_classical = 0.0015 * randn(size(t));
jitter_quantum = 0.0025 * randn(size(t));

% 3.3 Event-Based Disturbances (Spacecraft Maneuvers, Micro-meteoroid impacts)
% Spacecraft maneuver: Sudden acceleration changes around t = 2.5s and t = 3.8s
maneuver_disturbance = zeros(size(t));
maneuver_disturbance(t >= 2.4 & t <= 2.6) = 0.8 * sin(2*pi*8*(t(t >= 2.4 & t <= 2.6)));
maneuver_disturbance(t >= 3.7 & t <= 3.9) = 0.6 * sin(2*pi*10*(t(t >= 3.7 & t <= 3.9)));

% Micro-meteoroid impact: Random transient bursts
micro_impact = zeros(size(t));
num_impacts = 3;
for i = 1:num_impacts
    impact_time = 1 + rand() * 3;  % Random times between 1-4 seconds
    impact_duration = 0.05;  % 50ms impact
    impact_idx = find(t >= impact_time & t <= impact_time + impact_duration);
    if ~isempty(impact_idx)
        micro_impact(impact_idx) = 1.2 * exp(-3*(t(impact_idx) - impact_time) / impact_duration) .* ...
                                   sin(2*pi*20*(t(impact_idx) - impact_time));
    end
end

% 3.4 Total Environmental Disturbances
env_disturbance_classical = radiation_fault_classical + thermal_drift_classical + jitter_classical + maneuver_disturbance + micro_impact;
env_disturbance_quantum = radiation_fault_quantum + thermal_drift_quantum + jitter_quantum + maneuver_disturbance + 1.3*micro_impact;

% Apply disturbances to sensor outputs
classical_sensor_with_env = classical_sensor_output + env_disturbance_classical;
quantum_sensor_with_env = quantum_sensor_output + env_disturbance_quantum;

% Recalculate metrics after environmental disturbances
rms_classical_env = sqrt(mean((classical_sensor_with_env - signal_perfect).^2));
rms_quantum_env = sqrt(mean((quantum_sensor_with_env - signal_perfect).^2));

SNR_classical_env = 10 * log10(var(signal_perfect) / (var(classical_noise_total) + var(env_disturbance_classical)));
SNR_quantum_env = 10 * log10(var(signal_perfect) / (var(quantum_noise_total) + var(env_disturbance_quantum)));


%% ========================================================================
%% STEP 4: ELECTRONICS AND ADC MODELING
%% ========================================================================
fprintf('Step 4: Modeling electronics and ADC...\n')

% 4.1 Low-Noise Amplifier (LNA) Modeling
% LNA Gain
LNA_gain_classical = 20;  % 20 dB gain = 10x
LNA_gain_quantum = 15;    % 15 dB gain = 5.6x (quantum sensors use gentler gains)

% LNA 1/f (flicker) noise
LNA_flicker_classical = 0.01 * cumsum(randn(size(t))) / 100;
LNA_flicker_quantum = 0.015 * cumsum(randn(size(t))) / 100;

% LNA saturation effect (soft saturation)
V_sat = 3.3;  % Supply voltage
classical_lna_out = (LNA_gain_classical * classical_sensor_with_env);
quantum_lna_out = (LNA_gain_quantum * quantum_sensor_with_env);

% Apply soft saturation using tanh
classical_lna_out = V_sat * tanh(classical_lna_out / V_sat) + LNA_flicker_classical;
quantum_lna_out = V_sat * tanh(quantum_lna_out / V_sat) + LNA_flicker_quantum;

% 4.2 Anti-Aliasing Filter Design and Application
% Butterworth low-pass filter: cutoff at 200 Hz (half of Nyquist for 1000 Hz sampling)
filter_order = 4;
cutoff_freq = 200;  % Hz
nyquist_freq = Fs / 2;
normalized_cutoff = cutoff_freq / nyquist_freq;

% Design Butterworth filter
[b_butter, a_butter] = butter(filter_order, normalized_cutoff, 'low');

% Apply anti-aliasing filter
classical_filtered = filter(b_butter, a_butter, classical_lna_out);
quantum_filtered = filter(b_butter, a_butter, quantum_lna_out);

% 4.3 ADC Modeling - SAR (Successive Approximation Register) ADC
% SAR ADC parameters
bit_resolution = 12;  % 12-bit ADC
V_ref = 3.3;  % Reference voltage
quantization_levels = 2^bit_resolution;
LSB_classical = V_ref / quantization_levels;

% Sampling jitter (timing uncertainty) - affects SAR ADC
sampling_jitter_std = 1e-6;  % 1 microsecond std dev
jitter_samples = sampling_jitter_std * Fs * randn(size(t));

% Quantization noise (uniform distribution across LSB)
quantization_noise_classical = (LSB_classical/2) * (2*rand(size(t)) - 1);

% SAR ADC digitization
classical_adc_sar = round(classical_filtered / LSB_classical) * LSB_classical;
classical_adc_sar = classical_adc_sar + quantization_noise_classical;
classical_adc_sar = max(0, min(V_ref, classical_adc_sar));  % Clamp to valid range

% 4.4 ADC Modeling - Delta-Sigma (ΔΣ) ADC for Quantum Sensor
% ΔΣ ADC parameters (higher oversampling for quantum)
oversampling_ratio = 256;  % Typical oversampling ratio
bit_resolution_ds = 16;    % Effective bits with oversampling
Fs_oversampled = Fs * oversampling_ratio;  % Oversampled frequency = 256 kHz

% Upsample the filtered quantum signal for oversampled modulation
t_oversampled = 0:1/Fs_oversampled:T-1/Fs_oversampled;
quantum_filtered_upsampled = interp1(t, quantum_filtered, t_oversampled, 'linear', 'extrap');

% Realistic ΔΣ modulator: 2nd-order integrator + 1-bit quantizer with noise shaping
integrator1 = 0;
integrator2 = 0;
ds_bitstream = zeros(size(t_oversampled));

for i = 1:length(t_oversampled)
    % Input signal
    input_signal = quantum_filtered_upsampled(i);
    
    % First integrator stage
    if i == 1
        feedback = 0;
    else
        feedback = (ds_bitstream(i-1) - V_ref/2) * 2 / V_ref;  % Convert to [-1, 1]
    end
    
    error = input_signal - feedback;
    integrator1 = integrator1 + error;
    
    % Second integrator stage (adds noise shaping)
    integrator2 = integrator2 + integrator1;
    
    % 1-bit quantizer (comparator)
    if integrator2 > 0
        ds_bitstream(i) = V_ref;
    else
        ds_bitstream(i) = 0;
    end
end

% Decimation: Apply sinc filter and downsample back to original rate
% Sinc filter (moving average) acts as low-pass filter
sinc_filter_length = oversampling_ratio;  % Average over oversampling window
sinc_filter_coeffs = ones(1, sinc_filter_length) / sinc_filter_length;

% Apply sinc filter to bitstream
ds_filtered = filter(sinc_filter_coeffs, 1, ds_bitstream);

% Downsample back to original sample rate (take every 256th sample)
quantum_adc_ds = ds_filtered(1:oversampling_ratio:end);

% Ensure same length as original time vector
if length(quantum_adc_ds) > length(t)
    quantum_adc_ds = quantum_adc_ds(1:length(t));
elseif length(quantum_adc_ds) < length(t)
    quantum_adc_ds = [quantum_adc_ds, repmat(quantum_adc_ds(end), 1, length(t)-length(quantum_adc_ds))];
end

% Calculate effective bit resolution from ΔΣ (SNR improvement from oversampling)
% ENOB ≈ 1.76 + 0.5*log2(OSR) for 2nd-order ΔΣ
effective_bits_ds = 1.76 + 0.5 * log2(oversampling_ratio);  % ≈ 5.76 bits
fprintf('  ✓ ΔΣ Effective bits: %.2f (from noise shaping)\n', effective_bits_ds);

% Recalculate metrics after ADC
rms_classical_adc = sqrt(mean((classical_adc_sar - signal_perfect).^2));
rms_quantum_adc = sqrt(mean((quantum_adc_ds - signal_perfect).^2));

fprintf('  ✓ LNA: Classical Gain %.1f dB, Quantum Gain %.1f dB\n', ...
        20*log10(LNA_gain_classical), 20*log10(LNA_gain_quantum));
fprintf('  ✓ Anti-aliasing filter: %d-th order Butterworth @ %.0f Hz\n', filter_order, cutoff_freq);
fprintf('  ✓ SAR ADC: %d-bit, LSB = %.4f V\n', bit_resolution, LSB_classical);
fprintf('  ✓ ΔΣ ADC: Oversampling ratio = %d\n', oversampling_ratio);


%% ========================================================================
%% VISUALIZATION: COMPREHENSIVE ANALYSIS (STEPS 1-4)
%% ========================================================================
fprintf('Generating comprehensive visualization for Steps 1-4...\n')

figure('Color','w', 'Position', [50 50 1920 1200]);  % Larger figure size for 4 rows

% Define colors
color_truth = [0 0.6 0];
color_classical = [0 0.4 0.8];
color_quantum = [0.8 0.2 0.2];
color_env = [0.8 0.6 0];
color_adc = [0.6 0 0.8];

%% ========== ROW 1: SENSOR OUTPUTS (STEPS 1-2) ==========

% 1.1 Full time series
subplot(4,4,[1 2])
hold on
h1 = plot(t, signal_perfect, 'Color', color_truth, 'LineWidth', 2, 'DisplayName', 'Ground Truth');
h2 = plot(t, classical_sensor_output, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'Classical Sensor');
h3 = plot(t, quantum_sensor_output, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'Quantum Sensor');
xlabel('Time (s)', 'FontSize', 10, 'FontWeight', 'bold')
ylabel('Amplitude', 'FontSize', 10, 'FontWeight', 'bold')
title('A. Sensor Outputs (Steps 1-2)', 'FontSize', 11, 'FontWeight', 'bold')
legend([h1 h2 h3], 'Location', 'northeast', 'FontSize', 8)
grid on
xlim([0 T])
hold off

% 1.2 Zoomed view (normal sensors)
subplot(4,4,3)
t_zoom = 2.0; t_width = 0.8;
zoom_mask = (t >= t_zoom & t <= t_zoom + t_width);
hold on
plot(t(zoom_mask), signal_perfect(zoom_mask), 'Color', color_truth, 'LineWidth', 2.5)
plot(t(zoom_mask), classical_sensor_output(zoom_mask), 'Color', color_classical, 'LineWidth', 1.5)
plot(t(zoom_mask), quantum_sensor_output(zoom_mask), 'Color', color_quantum, 'LineWidth', 1.5)
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('B. Zoomed View (2.0-2.8s)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
xlim([t_zoom t_zoom+t_width])
hold off

% 1.3 SNR Comparison (before environment)
subplot(4,4,4)
snr_data = [SNR_classical_sensor, SNR_quantum_sensor];
bar_colors = [color_classical; color_quantum];
b = bar(snr_data, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Classical', 'Quantum'}, 'FontSize', 9, 'FontWeight', 'bold')
ylabel('SNR (dB)', 'FontSize', 9, 'FontWeight', 'bold')
title('C. SNR (Before Disturbances)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
ylim([0 max(snr_data)*1.2])
for i = 1:length(snr_data)
    text(i, snr_data(i)+0.3, sprintf('%.2f', snr_data(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8)
end

%% ========== ROW 2: ENVIRONMENTAL DISTURBANCES (STEP 3) ==========

% 2.1 Environmental disturbances - Classical
subplot(4,4,5)
hold on
plot(t, radiation_fault_classical, 'LineWidth', 0.8, 'DisplayName', 'Radiation')
plot(t, thermal_drift_classical, 'LineWidth', 0.8, 'DisplayName', 'Thermal Drift')
plot(t, jitter_classical, 'LineWidth', 0.8, 'DisplayName', 'Jitter')
plot(t, maneuver_disturbance, 'LineWidth', 1, 'DisplayName', 'Maneuver')
plot(t, micro_impact, 'LineWidth', 1, 'DisplayName', 'Micro-Impact')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Disturbance Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('D. Environmental Disturbances (Classical)', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'northeast', 'FontSize', 7)
grid on
xlim([0 T])
hold off

% 2.2 Environmental disturbances - Quantum
subplot(4,4,6)
hold on
plot(t, radiation_fault_quantum, 'LineWidth', 0.8, 'DisplayName', 'Radiation')
plot(t, thermal_drift_quantum, 'LineWidth', 0.8, 'DisplayName', 'Thermal Drift')
plot(t, jitter_quantum, 'LineWidth', 0.8, 'DisplayName', 'Jitter')
plot(t, maneuver_disturbance, 'LineWidth', 1, 'DisplayName', 'Maneuver')
plot(t, 1.3*micro_impact, 'LineWidth', 1, 'DisplayName', 'Micro-Impact')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Disturbance Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('E. Environmental Disturbances (Quantum)', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'northeast', 'FontSize', 7)
grid on
xlim([0 T])
hold off

% 2.3 Sensor outputs after environment
subplot(4,4,7)
hold on
plot(t, classical_sensor_with_env, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'Classical + Env')
plot(t, quantum_sensor_with_env, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'Quantum + Env')
plot(t, signal_perfect, 'Color', color_truth, 'LineWidth', 1.5, 'DisplayName', 'Ground Truth')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('F. Sensors with Environmental Impact', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'best', 'FontSize', 7)
grid on
xlim([0 T])
hold off

% 2.4 SNR degradation
subplot(4,4,8)
snr_before = [SNR_classical_sensor, SNR_quantum_sensor];
snr_after = [SNR_classical_env, SNR_quantum_env];
x = 1:2;
bar_width = 0.35;
hold on
bar(x - bar_width/2, snr_before, bar_width, 'FaceColor', color_classical, 'DisplayName', 'Before')
bar(x + bar_width/2, snr_after, bar_width, 'FaceColor', color_env, 'DisplayName', 'After')
set(gca, 'XTickLabel', {'Classical', 'Quantum'}, 'FontSize', 9, 'FontWeight', 'bold')
ylabel('SNR (dB)', 'FontSize', 9, 'FontWeight', 'bold')
title('G. SNR Degradation from Environment', 'FontSize', 10, 'FontWeight', 'bold')
legend('FontSize', 8)
grid on
hold off

%% ========== ROW 3: ELECTRONICS & LNA (STEP 4) ==========

% 3.1 LNA Output - Classical
subplot(4,4,9)
hold on
plot(t, classical_lna_out, 'Color', color_classical, 'LineWidth', 1.2)
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('H. LNA Output (Classical)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
xlim([0 T])
yline(V_sat, 'r--', 'LineWidth', 1, 'DisplayName', 'Saturation');
yline(-V_sat, 'r--', 'LineWidth', 1);
hold off

% 3.2 LNA Output - Quantum
subplot(4,4,10)
hold on
plot(t, quantum_lna_out, 'Color', color_quantum, 'LineWidth', 1.2)
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('I. LNA Output (Quantum)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
xlim([0 T])
yline(V_sat, 'r--', 'LineWidth', 1);
yline(-V_sat, 'r--', 'LineWidth', 1);
hold off

% 3.3 Filtered signal - Classical
subplot(4,4,11)
hold on
plot(t, classical_filtered, 'Color', color_classical, 'LineWidth', 1.2)
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('J. After Anti-Aliasing Filter (Classical)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
xlim([0 T])
hold off

% 3.4 Filtered signal - Quantum
subplot(4,4,12)
hold on
plot(t, quantum_filtered, 'Color', color_quantum, 'LineWidth', 1.2)
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('K. After Anti-Aliasing Filter (Quantum)', 'FontSize', 10, 'FontWeight', 'bold')
grid on
xlim([0 T])
hold off

%% ========== ROW 4: ADC OUTPUTS & METRICS ==========

% 4.1 SAR ADC Output (Classical)
subplot(4,4,13)
hold on
plot(t, classical_adc_sar, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'SAR ADC Output')
plot(t, signal_perfect, 'Color', color_truth, 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Ground Truth')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('L. SAR ADC Output (Classical)', 'FontSize', 10, 'FontWeight', 'bold')
legend('FontSize', 7)
grid on
xlim([0 T])
hold off

% 4.2 ΔΣ ADC Output (Quantum)
subplot(4,4,14)
hold on
plot(t, quantum_adc_ds, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'ΔΣ ADC Output')
plot(t, signal_perfect, 'Color', color_truth, 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Ground Truth')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Amplitude (V)', 'FontSize', 9, 'FontWeight', 'bold')
title('M. ΔΣ ADC Output (Quantum)', 'FontSize', 10, 'FontWeight', 'bold')
legend('FontSize', 7)
grid on
xlim([0 T])
hold off

% 4.3 RMS Error Progression
subplot(4,4,15)
stages = categorical({'Sensor', 'With Env', 'After ADC'});
stages = reordercats(stages, {'Sensor', 'With Env', 'After ADC'});
rms_classical_stages = [rms_classical; rms_classical_env; rms_classical_adc];
rms_quantum_stages = [rms_quantum; rms_quantum_env; rms_quantum_adc];

x = 1:3;
bar_width = 0.35;
hold on
b1 = bar(x - bar_width/2, rms_classical_stages, bar_width, 'FaceColor', color_classical, 'DisplayName', 'Classical');
b2 = bar(x + bar_width/2, rms_quantum_stages, bar_width, 'FaceColor', color_quantum, 'DisplayName', 'Quantum');
set(gca, 'XTickLabel', {'Sensor', 'With Env', 'After ADC'}, 'FontSize', 8, 'FontWeight', 'bold')
ylabel('RMS Error', 'FontSize', 9, 'FontWeight', 'bold')
title('N. RMS Error Progression', 'FontSize', 10, 'FontWeight', 'bold')
legend('FontSize', 8, 'Location', 'northwest')
grid on
hold off

% 4.4 Summary statistics table
subplot(4,4,16)
axis off
hold on

summary_lines = {
    '\bfSteps 1-4 Performance Summary'
    '────────────────────────────────'
    ''
    '\bfStep 1-2: Initial Sensing'
    sprintf('  Classical SNR: %.2f dB | RMS: %.4f', SNR_classical_sensor, rms_classical)
    sprintf('  Quantum SNR:   %.2f dB | RMS: %.4f', SNR_quantum_sensor, rms_quantum)
    ''
    '\bfStep 3: Environmental Impact'
    sprintf('  Classical: SNR ↓ %.2f dB', SNR_classical_sensor - SNR_classical_env)
    sprintf('  Quantum: SNR ↓ %.2f dB', SNR_quantum_sensor - SNR_quantum_env)
    ''
    '\bfStep 4: After ADC'
    sprintf('  Classical: RMS = %.4f', rms_classical_adc)
    sprintf('  Quantum: RMS = %.4f', rms_quantum_adc)
    ''
    '\bfKey Insights:'
    '  • Radiation: 2% (C), 4% (Q)'
    '  • Thermal drift higher for Q'
    '  • SAR vs ΔΣ ADC tradeoff'
};

text(0.05, 0.98, summary_lines, 'FontSize', 7, ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
     'FontName', 'FixedWidth', 'Interpreter', 'tex')

rectangle('Position', [0 0 1 1], 'EdgeColor', 'k', 'LineWidth', 1.5)
xlim([0 1]); ylim([0 1]);
hold off

%% Main title
sgtitle('STEPS 1-4: Ground Truth → Sensor Modeling → Environmental Disturbances → Electronics & ADC', ...
        'FontSize', 13, 'FontWeight', 'bold')

%% Save figure with high resolution
saveas(gcf, 'steps_1_to_4_complete.png')
print(gcf, 'steps_1_to_4_complete_highres.png', '-dpng', '-r300')  % High-res version

fprintf('✓ Figure saved: steps_1_to_4_complete.png\n')
fprintf('✓ High-res figure saved: steps_1_to_4_complete_highres.png\n')


%% ========================================================================
%% DATA EXPORT: CSV GENERATION
%% ========================================================================
fprintf('Exporting data to CSV...\n')

% 1. Export Summary Statistics (all steps)
Metric = {'SNR (dB) - Step 1-2'; 'SNR (dB) - Step 3 (w/ Env)'; 'RMS Error - Step 1-2'; ...
          'RMS Error - Step 3 (w/ Env)'; 'RMS Error - Step 4 (ADC)'; 'Noise Std Dev (sigma)'};
Classical = [SNR_classical_sensor; SNR_classical_env; rms_classical; rms_classical_env; ...
             rms_classical_adc; std(classical_noise_total)];
Quantum = [SNR_quantum_sensor; SNR_quantum_env; rms_quantum; rms_quantum_env; ...
           rms_quantum_adc; std(quantum_noise_total)];

summary_table = table(Metric, Classical, Quantum);
writetable(summary_table, 'sensor_performance_summary.csv');

% 2. Export Detailed Time-Series Data (Steps 1-4)
sensor_data_table = table(t', signal_perfect', classical_sensor_output', quantum_sensor_output', ...
    classical_sensor_with_env', quantum_sensor_with_env', classical_adc_sar', quantum_adc_ds', ...
    'VariableNames', {'Time_s', 'Ground_Truth', 'Classical_Sensor_Steps_1_2', 'Quantum_Sensor_Steps_1_2', ...
                      'Classical_w_Env_Step_3', 'Quantum_w_Env_Step_3', 'Classical_ADC_SAR_Step_4', 'Quantum_ADC_DS_Step_4'});

writetable(sensor_data_table, 'sensor_raw_data_steps_1_4_complete.csv');

% 3. Export Environmental Disturbances (Step 3 Details)
env_disturbance_table = table(t', radiation_fault_classical', radiation_fault_quantum', ...
    thermal_drift_classical', thermal_drift_quantum', jitter_classical', jitter_quantum', ...
    maneuver_disturbance', micro_impact', env_disturbance_classical', env_disturbance_quantum', ...
    'VariableNames', {'Time_s', 'Radiation_Classical', 'Radiation_Quantum', ...
                      'ThermalDrift_Classical', 'ThermalDrift_Quantum', 'Jitter_Classical', 'Jitter_Quantum', ...
                      'Maneuver_Disturbance', 'Micro_Impact', 'Total_Env_Classical', 'Total_Env_Quantum'});

writetable(env_disturbance_table, 'environmental_disturbances_step_3.csv');

% 4. Export Electronics/ADC Details (Step 4 Details)
adc_data_table = table(t', classical_lna_out', quantum_lna_out', classical_filtered', quantum_filtered', ...
    classical_adc_sar', quantum_adc_ds', ...
    'VariableNames', {'Time_s', 'Classical_LNA_Output', 'Quantum_LNA_Output', ...
                      'Classical_Filtered', 'Quantum_Filtered', 'Classical_SAR_ADC', 'Quantum_DS_ADC'});

writetable(adc_data_table, 'electronics_adc_step_4.csv');

fprintf('✓ CSV Saved: sensor_performance_summary.csv\n')
fprintf('✓ CSV Saved: sensor_raw_data_steps_1_4_complete.csv\n')
fprintf('✓ CSV Saved: environmental_disturbances_step_3.csv\n')
fprintf('✓ CSV Saved: electronics_adc_step_4.csv\n')
fprintf('\n✓ Steps 1-4 COMPLETE!\n')