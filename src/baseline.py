from __future__ import annotations
import numpy as np

from preprocessing import stft, istft, magnitude_phase, to_complex

def spectral_subtraction(noisy: np.ndarray, over_subtraction: float = 1.5,
                         spectral_floor: float = 0.05) -> np.ndarray:
    """Sustracción espectral clásica: |X| = max(|Y| - alpha*|D|, beta*|Y|)."""
    S = stft(noisy)
    mag, phase = magnitude_and_phase(S)
    noise_mag = estimate_noise_magnitude(mag)

    clean_mag = mag - over_subtraction * noise_mag
    floor = spectral_floor * mag
    clean_mag = np.maximum(clean_mag, floor)

    S_clean = to_complex(clean_mag, phase)
    return istft(S_clean, length=len(noisy)).astype(np.float32)

def spectral_gating(noisy: np.ndarray, n_std: float = 1.5,
                    gain_floor: float = 0.05, smooth: bool = True) -> np.ndarray:
    """Compuerta espectral (spectral gating): baseline clásico basado en máscara.

    Estima un perfil de ruido por banda (media y desviación de los fotogramas más
    silenciosos), fija un umbral T(k) = mu + n_std*sigma, y construye una máscara:
    deja pasar los puntos por encima del umbral (voz) y atenúa los de abajo (ruido)
    hasta una ganancia mínima gain_floor. Opcionalmente suaviza la máscara.
    """
    S = stft(noisy)
    mag, phase = magnitude_phase(S)

    p20 = np.quantile(mag, 0.20, axis=1, keepdims=True)
    quiet = np.where(mag <= p20, mag, np.nan)
    noise_mean = np.nanmean(quiet, axis=1, keepdims=True)
    noise_std = np.nanstd(quiet, axis=1, keepdims=True)
    threshold = noise_mean + n_std * noise_std

    mask = (mag >= threshold).astype(np.float32)
    mask = mask + (1.0 - mask) * gain_floor
    if smooth:
        mask = _smooth2d(mask, k=3).astype(np.float32)
        mask = np.clip(mask, gain_floor, 1.0)

    S_clean = to_complex(mag * mask, phase)
    return istft(S_clean, length=len(noisy)).astype(np.float32)

def _smooth2d(m: np.ndarray, k: int = 3) -> np.ndarray:
    """Suavizado separable (media móvil) en frecuencia y tiempo, solo con numpy."""
    ker = np.ones(k) / k
    m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 1, m)
    m = np.apply_along_axis(lambda r: np.convolve(r, ker, mode="same"), 0, m)
    return m

def estimate_noise_magnitude(mag: np.ndarray, quantile: float = 0.10) -> np.ndarray:
    """Estima la magnitud del ruido por banda tomando un percentil bajo en el tiempo."""
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
    if not clean_wavs:
        raise SystemExit(f"Coloca WAV reales en {root} para ejecutar la prueba.")
    name0 = clean_wavs[0].name
    clean = load_audio(root / "clean_testset" / name0)
    noisy = load_audio(root / "noisy_testset" / name0)
    clean, noisy = match_length(clean, noisy)
    print(f"Archivo: {name0}")
    for name, fn in [("spectral gating", spectral_gating),
                     ("sustraccion espectral", spectral_subtraction)]:
        est = fn(noisy)[:len(clean)]
        res_noise = np.sqrt(np.mean((clean - est) ** 2))
        print(f"{name}: RMS(resultado vs limpio) = {res_noise:.4f}")