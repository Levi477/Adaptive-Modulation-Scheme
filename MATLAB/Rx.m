clc; clear; close all;

%% 1. Load NI DAQ Data
rx_passband = dlmread('Acquired_data.csv');
rx_passband = rx_passband(:,1); % Ensure 1D

%% 2. Digital Downconversion (Extract I and Q from 1 Wire)
F_sample = 4e6;
f_sub = 500e3;
t = (0:length(rx_passband)-1)' / F_sample;

% Multiply by negative carrier to bring baseband down to 0 Hz
rx_mixed = rx_passband .* exp(-1j * 2 * pi * f_sub * t);

% Low Pass Filter to remove double-frequency terms
lpf = designfilt('lowpassfir', 'PassbandFrequency', 300e3, 'StopbandFrequency', 600e3, 'SampleRate', F_sample);
rx_baseband = filter(lpf, rx_mixed);

%% 3. Matched Filtering
sps = 8;
h_rrc_rx = comm.RaisedCosineReceiveFilter('RolloffFactor', 0.35, 'FilterSpanInSymbols', 6, 'InputSamplesPerSymbol', sps);
rx_filt = step(h_rrc_rx, rx_baseband);

%% 4. Synchronization (Look for BPSK Sync)
h_pn = comm.PNSequence('Polynomial',[6 5 0], 'SamplesPerFrame', 64, 'InitialConditions',[0 0 0 0 0 1]);
sync_bits = step(h_pn);
sync_syms = pskmod(sync_bits, 2, 'InputType', 'bit');
sync_upsampled = upsample(sync_syms, sps);

[cor, lags] = xcorr(rx_filt, sync_upsampled);
[~, idx] = max(abs(cor));
start_sample = lags(idx) + 1;
rx_synced = rx_filt(start_sample : end);
rx_syms = rx_synced(1:sps:end); % Downsample

%% 5. Read Header & Determine Scheme
% The first 64 are sync. The next 16 are the scheme header.
header_syms_rx = rx_syms(65 : 64+16);
header_bits_rx = pskdemod(header_syms_rx, 2, 'OutputType', 'bit');

% Convert 16 bits back to integer scheme index
scheme_idx = bin2dec(num2str(header_bits_rx'));
schemes = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM"];
M_vals = [2, 4, 8, 16, 64, 256];

detected_scheme = schemes(scheme_idx + 1);
M = M_vals(scheme_idx + 1);
fprintf('Receiver Read Header. Adaptive Scheme is: %s\n', detected_scheme);

%% 6. Demodulate Payload dynamically
payload_syms_rx = rx_syms(64 + 16 + 1 : end);

if contains(detected_scheme, 'PSK')
    payload_bits_rx = pskdemod(payload_syms_rx, M, 'OutputType', 'bit');
else
    payload_bits_rx = qamdemod(payload_syms_rx, M, 'OutputType', 'bit', 'UnitAveragePower', true);
end

% Plot Constellation to visually verify
scatterplot(payload_syms_rx); title(['Received Payload: ', char(detected_scheme)]);
