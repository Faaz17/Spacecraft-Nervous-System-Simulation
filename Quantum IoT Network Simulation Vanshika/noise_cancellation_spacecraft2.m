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
%% VISUALIZATION: COMPREHENSIVE ANALYSIS
%% ========================================================================
fprintf('Generating comprehensive visualization...\n')

figure('Color','w', 'Position', [50 50 1920 1080]);  % Larger figure size

% Define colors
color_truth = [0 0.6 0];
color_classical = [0 0.4 0.8];
color_quantum = [0.8 0.2 0.2];

%% ========== ROW 1: SENSOR OUTPUTS ==========

% 1.1 Full time series
subplot(3,4,[1 2])
hold on
h1 = plot(t, signal_perfect, 'Color', color_truth, 'LineWidth', 2, 'DisplayName', 'Ground Truth');
h2 = plot(t, classical_sensor_output, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'Classical Sensor');
h3 = plot(t, quantum_sensor_output, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'Quantum Sensor');
xlabel('Time (s)', 'FontSize', 10, 'FontWeight', 'bold')
ylabel('Amplitude', 'FontSize', 10, 'FontWeight', 'bold')
title('A. Sensor Outputs - Full Signal (5 seconds)', 'FontSize', 11, 'FontWeight', 'bold')
legend([h1 h2 h3], 'Location', 'northeast', 'FontSize', 9)
grid on
xlim([0 T])
ylim([min([signal_perfect classical_sensor_output quantum_sensor_output])-0.2, ...
      max([signal_perfect classical_sensor_output quantum_sensor_output])+0.2])
hold off

% 1.2 Zoomed view
subplot(3,4,3)
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

% 1.3 SNR Comparison
subplot(3,4,4)
snr_data = [SNR_classical_sensor, SNR_quantum_sensor];
bar_colors = [color_classical; color_quantum];
b = bar(snr_data, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Classical', 'Quantum'}, 'FontSize', 10, 'FontWeight', 'bold')
ylabel('SNR (dB)', 'FontSize', 10, 'FontWeight', 'bold')
title('C. Signal-to-Noise Ratio', 'FontSize', 10, 'FontWeight', 'bold')
grid on
ylim([0 max(snr_data)*1.2])
% Add value labels
for i = 1:length(snr_data)
    text(i, snr_data(i)+0.3, sprintf('%.2f dB', snr_data(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 10)
end

%% ========== ROW 2: NOISE ANALYSIS ==========

% 2.1 Classical noise components
subplot(3,4,5)
hold on
plot(t, classical_thermal, 'LineWidth', 1, 'DisplayName', 'Thermal')
plot(t, classical_flicker, 'LineWidth', 1, 'DisplayName', '1/f Noise')
plot(t, classical_drift, 'LineWidth', 1, 'DisplayName', 'Drift')
plot(t, classical_radiation, 'LineWidth', 1, 'DisplayName', 'Radiation')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Noise Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('D. Classical Sensor: Noise Breakdown', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'northeast', 'FontSize', 7)
grid on
xlim([0 T])
hold off

% 2.2 Quantum noise components
subplot(3,4,6)
hold on
plot(t, quantum_phase_noise, 'LineWidth', 1, 'DisplayName', 'Phase Noise')
plot(t, quantum_decoherence, 'LineWidth', 1, 'DisplayName', 'Decoherence')
plot(t, environmental_B_noise, 'LineWidth', 1, 'DisplayName', 'Env. Sensitivity')
plot(t, quantum_init_errors, 'LineWidth', 1, 'DisplayName', 'Init Errors')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Noise Amplitude', 'FontSize', 9, 'FontWeight', 'bold')
title('E. Quantum Sensor: Noise Breakdown', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'northeast', 'FontSize', 7)
grid on
xlim([0 T])
hold off

% 2.3 Total noise comparison (time domain)
subplot(3,4,7)
hold on
plot(t, classical_noise_total, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'Classical Noise')
plot(t, quantum_noise_total, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'Quantum Noise')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Total Noise', 'FontSize', 9, 'FontWeight', 'bold')
title('F. Total Noise Comparison', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'best', 'FontSize', 8)
grid on
xlim([0 T])
hold off

% 2.4 Measurement errors
subplot(3,4,8)
classical_error = classical_sensor_output - signal_perfect;
quantum_error = quantum_sensor_output - signal_perfect;
hold on
plot(t, classical_error, 'Color', color_classical, 'LineWidth', 1.2, 'DisplayName', 'Classical Error')
plot(t, quantum_error, 'Color', color_quantum, 'LineWidth', 1.2, 'DisplayName', 'Quantum Error')
xlabel('Time (s)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Measurement Error', 'FontSize', 9, 'FontWeight', 'bold')
title('G. Measurement Errors', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'best', 'FontSize', 8)
grid on
xlim([0 T])
yline(0, 'k--', 'LineWidth', 0.5)
hold off

%% ========== ROW 3: FREQUENCY DOMAIN ANALYSIS ==========

% 3.1 Signal power spectral density
subplot(3,4,9)
[Pxx_truth, f_truth] = pwelch(signal_perfect, hamming(1024), 512, 2048, Fs);
[Pxx_classical, f_classical] = pwelch(classical_sensor_output, hamming(1024), 512, 2048, Fs);
[Pxx_quantum, f_quantum] = pwelch(quantum_sensor_output, hamming(1024), 512, 2048, Fs);
semilogy(f_truth, Pxx_truth, 'Color', color_truth, 'LineWidth', 2, 'DisplayName', 'Ground Truth')
hold on
semilogy(f_classical, Pxx_classical, 'Color', color_classical, 'LineWidth', 1.5, 'DisplayName', 'Classical')
semilogy(f_quantum, Pxx_quantum, 'Color', color_quantum, 'LineWidth', 1.5, 'DisplayName', 'Quantum')
xlabel('Frequency (Hz)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('PSD', 'FontSize', 9, 'FontWeight', 'bold')
title('H. Frequency Domain: Sensor Outputs', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'southwest', 'FontSize', 7)
grid on
xlim([0 200])
hold off

% 3.2 Noise power spectral density
subplot(3,4,10)
[Pxx_noise_classical, f_nc] = pwelch(classical_noise_total, hamming(1024), 512, 2048, Fs);
[Pxx_noise_quantum, f_nq] = pwelch(quantum_noise_total, hamming(1024), 512, 2048, Fs);
semilogy(f_nc, Pxx_noise_classical, 'Color', color_classical, 'LineWidth', 2, 'DisplayName', 'Classical Noise')
hold on
semilogy(f_nq, Pxx_noise_quantum, 'Color', color_quantum, 'LineWidth', 2, 'DisplayName', 'Quantum Noise')
xlabel('Frequency (Hz)', 'FontSize', 9, 'FontWeight', 'bold')
ylabel('Noise PSD', 'FontSize', 9, 'FontWeight', 'bold')
title('I. Frequency Domain: Noise', 'FontSize', 10, 'FontWeight', 'bold')
legend('Location', 'southwest', 'FontSize', 7)
grid on
xlim([0 200])
hold off

% 3.3 RMS Error Statistics
subplot(3,4,11)
rms_classical = sqrt(mean(classical_error.^2));
rms_quantum = sqrt(mean(quantum_error.^2));
rms_data = [rms_classical, rms_quantum];
b = bar(rms_data, 'FaceColor', 'flat');
b.CData = bar_colors;
set(gca, 'XTickLabel', {'Classical', 'Quantum'}, 'FontSize', 9, 'FontWeight', 'bold')
ylabel('RMS Error', 'FontSize', 9, 'FontWeight', 'bold')
title('J. RMS Error', 'FontSize', 10, 'FontWeight', 'bold')
grid on
ylim([0 max(rms_data)*1.15])
for i = 1:length(rms_data)
    text(i, rms_data(i)+0.005, sprintf('%.4f', rms_data(i)), ...
         'HorizontalAlignment', 'center', 'FontWeight', 'bold', 'FontSize', 8)
end

% 3.4 Summary statistics table
% 3.4 Summary statistics table
subplot(3,4,12)
axis off
hold on

% COMPACT SUMMARY: Combining lines to ensure it fits the vertical space
summary_lines = {
    '\bfSensor Performance'
    '──────────────────'
    '\bfClassical Sensor:'
    sprintf('  SNR: %.2f dB | RMS: %.4f', SNR_classical_sensor, rms_classical)
    '\bfQuantum Sensor:'
    sprintf('  SNR: %.2f dB | RMS: %.4f', SNR_quantum_sensor, rms_quantum)
    '\bfKey Findings:'
    sprintf('  \\Delta SNR: +%.2f dB', SNR_quantum_sensor - SNR_classical_sensor)
    '  - Classical: Stable'
    '  - Quantum: High Sensitivity'
    '  - Issue: Decoherence'
};

% Display text with Font 7 and start it higher (0.98)
text(0.05, 0.98, summary_lines, 'FontSize', 7, ...
     'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', ...
     'FontName', 'FixedWidth', 'Interpreter', 'tex')

% Force axis limits to [0 1] and draw the border
rectangle('Position', [0 0 1 1], 'EdgeColor', 'k', 'LineWidth', 1.5)
xlim([0 1]); ylim([0 1]);
hold off
%% Main title with reduced font size
sgtitle('STEPS 1 & 2: Ground Truth + Classical vs Quantum Sensor Modeling', ...
        'FontSize', 12, 'FontWeight', 'bold')

%% Adjust spacing
set(gcf, 'Units', 'normalized')
% Tighten subplot spacing
set(gcf, 'DefaultAxesPosition', [0.1, 0.1, 0.8, 0.8])

%% Save figure with high resolution
saveas(gcf, 'steps_1_and_2_complete.png')
print(gcf, 'steps_1_and_2_complete_highres.png', '-dpng', '-r300')  % High-res version

fprintf('✓ Figure saved: steps_1_and_2_complete.png\n')
fprintf('✓ High-res figure saved: steps_1_and_2_complete_highres.png\n')


%% ========================================================================
%% DATA EXPORT: CSV GENERATION
%% ========================================================================
fprintf('Exporting data to CSV...\n')

% 1. Export Summary Statistics
Metric = {'SNR (dB)'; 'RMS Error'; 'Noise Std Dev (sigma)'};
Classical = [SNR_classical_sensor; rms_classical; std(classical_noise_total)];
Quantum = [SNR_quantum_sensor; rms_quantum; std(quantum_noise_total)];

summary_table = table(Metric, Classical, Quantum);
writetable(summary_table, 'sensor_performance_summary.csv');

% 2. Export Raw Time-Series Data (for Excel/External analysis)
% We ensure all vectors are column vectors using the (') transpose
sensor_data_table = table(t', signal_perfect', classical_sensor_output', quantum_sensor_output', ...
    'VariableNames', {'Time_s', 'Ground_Truth', 'Classical_Output', 'Quantum_Output'});

writetable(sensor_data_table, 'sensor_raw_data_steps_1_2.csv');

fprintf('✓ CSV Saved: sensor_performance_summary.csv\n')
fprintf('✓ CSV Saved: sensor_raw_data_steps_1_2.csv\n')