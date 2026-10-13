from __future__ import annotations
import numpy as np
import librosa

SAMPLE_RATE = 16_000

def stft(y: np.ndarray) -> np.ndarray:
    return librosa.stft(y, n_fft = 512)

def istft(S: np.ndarray, length: int | None = None) -> np.ndarray:
    return librosa.istft(S, length=length)

def magnitude_phase(S: np.ndarray):
    return np.abs(S), np.angle(S)

def to_complex(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
    return magnitude * np.exp(1j * phase)

def reconstruction_error(y: np.ndarray) -> float:
    y_rec = istft(stft(y), length=len(y))
    return float(np.sqrt(np.mean((y - y_rec) ** 2)))

def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32)

def match_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    default = Path(__file__).resolve().parents[1] / "data" / "raw" / "voicebank" / "clean_testset"
    wavs = list(default.glob("*.wav"))
    x = load_audio(wavs[0])
    err = reconstruction_error(x)
    print(f"Archivo: {wavs[0].name}")
    print(f"Error de reconstruccion STFT/ISTFT (RMS): {err:.2e}")