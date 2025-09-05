import io
import math
import os
import random
import time
from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import requests
import soundfile as sf
from pydub import AudioSegment


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def download_audio_to_wav(url: str, target_wav_path: str) -> str:
    """Download audio from URL and save as WAV. Returns saved path."""
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    # Try load with pydub to handle various formats, then export to WAV
    audio = AudioSegment.from_file(io.BytesIO(resp.content))
    audio.export(target_wav_path, format="wav")
    return target_wav_path


def load_audio_for_features(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    y, _sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def compute_audio_features(wav_path: str) -> Dict[str, float]:
    y, sr = load_audio_for_features(wav_path)
    if y.size == 0:
        return {
            "duration_seconds": 0.0,
            "tempo_bpm": 0.0,
            "rms_mean": 0.0,
            "spectral_centroid_mean": 0.0,
            "zero_crossing_rate_mean": 0.0,
            "mfcc1_mean": 0.0,
        }

    duration_seconds = float(librosa.get_duration(y=y, sr=sr))

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(tempo)
    except Exception:
        tempo_bpm = 0.0

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = float(np.mean(rms))

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = float(np.mean(spectral_centroid))

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = float(np.mean(zcr))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc1_mean = float(np.mean(mfcc[0]))

    return {
        "duration_seconds": duration_seconds,
        "tempo_bpm": tempo_bpm,
        "rms_mean": rms_mean,
        "spectral_centroid_mean": spectral_centroid_mean,
        "zero_crossing_rate_mean": zero_crossing_rate_mean,
        "mfcc1_mean": mfcc1_mean,
    }


def make_snippet(
    source_wav: str,
    target_wav: str,
    snippet_ms: int = 15000,
    fade_ms: int = 500,
    method: str = "random",
    seed: Optional[int] = None,
) -> Tuple[str, str, float]:
    """Create a snippet from source_wav, save to target_wav.

    Returns (snippet_path, snippet_method, snippet_length_seconds).
    """
    audio = AudioSegment.from_file(source_wav)
    length_ms = len(audio)
    rng = random.Random(seed)

    # If shorter than desired snippet, loop to reach length
    if length_ms < snippet_ms:
        loops = math.ceil(snippet_ms / max(1, length_ms))
        audio = audio * loops
        length_ms = len(audio)

    if method == "random":
        max_start = max(0, length_ms - snippet_ms)
        start_ms = rng.randint(0, max_start) if max_start > 0 else 0
        snippet = audio[start_ms : start_ms + snippet_ms]
    elif method == "highest_rms":
        # Choose the window with highest RMS energy using librosa for analysis
        start_ms = _find_highest_rms_window_ms(source_wav, snippet_ms)
        start_ms = int(max(0, min(start_ms, max(0, length_ms - snippet_ms))))
        snippet = audio[start_ms : start_ms + snippet_ms]
    else:
        # Default fallback behaves like random
        max_start = max(0, length_ms - snippet_ms)
        start_ms = rng.randint(0, max_start) if max_start > 0 else 0
        snippet = audio[start_ms : start_ms + snippet_ms]

    # Apply fades
    snippet = snippet.fade_in(fade_ms).fade_out(fade_ms)

    ensure_dir(os.path.dirname(target_wav))
    snippet.export(target_wav, format="wav")
    return target_wav, method, snippet_ms / 1000.0


def _find_highest_rms_window_ms(path: str, window_ms: int, sr: int = 22050) -> int:
    """Return the start position (ms) of the highest-RMS window of given length.

    Uses a hop of 50ms for efficiency.
    """
    y, _sr = librosa.load(path, sr=sr, mono=True)
    if y.size == 0:
        return 0
    window_samples = int(sr * (window_ms / 1000.0))
    hop_samples = int(sr * 0.05)
    if window_samples <= 0 or window_samples >= len(y):
        return 0
    best_start = 0
    best_rms = -1.0
    for start in range(0, len(y) - window_samples + 1, hop_samples):
        segment = y[start : start + window_samples]
        # Root-mean-square energy
        rms = float(np.sqrt(np.mean(segment**2))) if segment.size else 0.0
        if rms > best_rms:
            best_rms = rms
            best_start = start
    return int(1000.0 * best_start / sr)


