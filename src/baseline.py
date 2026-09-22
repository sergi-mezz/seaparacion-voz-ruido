from __future__ import annotations
import numpy as np

from preprocessing import stft, istft

def spectral_subtraction(noisy: np.ndarray, over_subtraction: float = 1.5,
                         spectral_floor: float = 0.05) -> np.ndarray:
    S = stft(noisy)
    mag, phase = magnitude_and_phase(S)
    noise_mag = estimate_noise_magnitude(mag)

    clean_mag = mag - over_subtraction * noise_mag
    floor = spectral_floor * mag
    clean_mag = np.maximum(clean_mag, floor)

    S_clean = to_complex(clean_mag, phase)
    return istft(S_clean, length=len(noisy)).astype(np.float32)

def estimate_noise_magnitude(mag: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    return np.quantile(mag, quantile, axis=1, keepdims=True)

def magnitude_and_phase(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.abs(S), np.angle(S)

def to_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * phase)

def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

if __name__ == "__main__":

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing import load_audio

    root = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank"
    clean_wavs = sorted((root / "clean_testset").glob("*.wav")) if (root / "clean_testset").exists() else []
    name0 = clean_wavs[0].name
    clean = load_audio(root / "clean_testset" / name0)
    noisy = load_audio(root / "noisy_testset" / name0)
    clean, noisy = match_length(clean, noisy)
    print(f"Archivo: {name0}")
    est = spectral_subtraction(noisy)[:len(clean)]
    res_noise = np.sqrt(np.mean((clean - est) ** 2))
    print(f"RMS(residuo vs limpio) = {res_noise:.4f}")