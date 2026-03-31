import numpy as np
import pandas as pd

# ==========================================
# 1. MODULATION GENERATORS
# ==========================================


def generate_m_psk(M):
    angles = 2 * np.pi * np.arange(M) / M
    return np.exp(1j * angles)


def generate_m_qam(M):
    m_sqrt = int(np.sqrt(M))
    pam = 2 * np.arange(m_sqrt) - (m_sqrt - 1)
    x, y = np.meshgrid(pam, pam)
    constellation = (x + 1j * y).flatten()
    avg_power = np.mean(np.abs(constellation) ** 2)
    return constellation / np.sqrt(avg_power)


# Ordered by throughput (lowest to highest)
SCHEMES = {
    "BPSK": generate_m_psk(2),
    "QPSK": generate_m_psk(4),
    "8PSK": generate_m_psk(8),
    "16QAM": generate_m_qam(16),
    "64QAM": generate_m_qam(64),
    "256QAM": generate_m_qam(256),
}

# ==========================================
# 2. CORE SIMULATION ENGINE
# ==========================================


def simulate_environment(snr_db, tx_power_dbm, num_symbols=5000, target_ber=1e-3):
    """
    Generates ONE environmental snapshot (noise, fading) and tests all schemes against it.
    """
    tx_power_linear = 10 ** ((tx_power_dbm - 30) / 10)  # Convert dBm to Watts

    # --- 1. GENERATE SHARED ENVIRONMENTAL CONDITIONS ---
    # Rician Fading (Indoor LOS)
    K_linear = 10 ** (10 / 10)  # 10 dB K-factor
    mu = np.sqrt(K_linear / (2 * (K_linear + 1)))
    sigma = np.sqrt(1 / (2 * (K_linear + 1)))
    h = (sigma * np.random.randn(num_symbols) + mu) + 1j * (
        sigma * np.random.randn(num_symbols) + mu
    )
    mean_channel_gain = np.mean(np.abs(h))

    # Phase Noise (Oscillator Jitter)
    phase_noise_variance = 0.03
    phase_noise = np.exp(1j * np.random.normal(0, phase_noise_variance, num_symbols))

    # Carrier Frequency Offset
    cfo_hz = 150
    t = np.arange(num_symbols) / 1e6
    cfo_shift = np.exp(1j * 2 * np.pi * cfo_hz * t)

    # Base AWGN (Normalized to SNR)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = tx_power_linear / snr_linear
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(num_symbols) + 1j * np.random.randn(num_symbols)
    )

    # --- 2. TEST ALL SCHEMES IN THIS ENVIRONMENT ---
    results = {}

    for scheme_name, constellation in SCHEMES.items():
        M = len(constellation)
        bits_per_symbol = int(np.log2(M))

        # Transmitter
        tx_symbols_idx = np.random.randint(0, M, num_symbols)
        tx_symbols = constellation[tx_symbols_idx]
        tx_signal = tx_symbols * np.sqrt(tx_power_linear)

        # Apply the exact same channel and noise
        rx_signal_clean = tx_signal * h * phase_noise * cfo_shift
        rx_signal = rx_signal_clean + noise

        # Receiver (Perfect CFO correction, Zero-Forcing Equalization)
        rx_cfo_corrected = rx_signal * np.exp(-1j * 2 * np.pi * cfo_hz * t)
        rx_eq = rx_cfo_corrected / h
        rx_normalized = rx_eq / np.sqrt(tx_power_linear)

        # Demodulation
        rx_symbols_idx = np.zeros(num_symbols, dtype=int)
        for i in range(num_symbols):
            distances = np.abs(rx_normalized[i] - constellation)
            rx_symbols_idx[i] = np.argmin(distances)

        # Calculate BER
        symbol_errors = np.sum(tx_symbols_idx != rx_symbols_idx)
        ber = (symbol_errors / num_symbols) / bits_per_symbol

        results[scheme_name] = ber

    # --- 3. DETERMINE THE BEST SCHEME ---
    # We iterate from Highest Data Rate to Lowest. First one to beat target_ber wins.
    best_scheme = "BPSK"  # Default fallback if channel is terrible
    schemes_reversed = ["256QAM", "64QAM", "16QAM", "8PSK", "QPSK", "BPSK"]

    for sch in schemes_reversed:
        if results[sch] <= target_ber:
            best_scheme = sch
            break

    # Return the dataset row
    return {
        "Tx_Power_dBm": round(tx_power_dbm, 2),
        "Target_SNR_dB": round(snr_db, 2),
        "Mean_Channel_Gain": round(mean_channel_gain, 4),
        "Noise_Power_Watts": noise_power,
        "Phase_Noise_Var": phase_noise_variance,
        "BPSK_BER": results["BPSK"],
        "QPSK_BER": results["QPSK"],
        "8PSK_BER": results["8PSK"],
        "16QAM_BER": results["16QAM"],
        "64QAM_BER": results["64QAM"],
        "256QAM_BER": results["256QAM"],
        "Label_Best_Scheme": best_scheme,
    }


# ==========================================
# 3. LARGE DATASET GENERATION
# ==========================================

NUM_SAMPLES = 20000  # Number of independent environmental states to simulate
SYMBOLS_PER_TEST = 5000  # Enough to measure BER accurately

print(f"Starting simulation of {NUM_SAMPLES} unique channel environments...")
dataset = []

for i in range(NUM_SAMPLES):
    # Randomize conditions for each sample to create a rich dataset
    rand_snr = np.random.uniform(0, 40)  # SNR between 0 dB and 40 dB
    rand_tx_pow = np.random.uniform(10, 30)  # Tx Power between 10 dBm and 30 dBm

    row = simulate_environment(rand_snr, rand_tx_pow, num_symbols=SYMBOLS_PER_TEST)
    dataset.append(row)

    if (i + 1) % 1000 == 0:
        print(f"Generated {i + 1} / {NUM_SAMPLES} samples...")

# Compile into DataFrame
df = pd.DataFrame(dataset)

print("\nSimulation Complete. Class distribution of the 'Best Scheme':")
print(df["Label_Best_Scheme"].value_counts())

# Save to CSV
output_filename = "amc_dataset.csv"
df.to_csv(output_filename, index=False)
print(f"\nDataset successfully saved to '{output_filename}'.")
