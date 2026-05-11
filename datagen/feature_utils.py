import numpy as np

# Retained patch to prevent breaking librosa<0.10 if numpy>=1.24 is installed
try:
    if not hasattr(np, 'complex'):
        np.complex = complex
except Exception:
    pass

from cache_utils import get_cached_top_regions
import librosa

def find_top_audio_regions(
    audio: np.ndarray,
    sr: int,
    region_duration: float,
    top_k: int = 5,
    hop_duration: float = 0.5,
    track_id: str = None
) -> list:
    """
    Optimized version using global array calculations and moving averages.
    Identify top-k audio regions based on a composite score of multiple features.
    """
    assert audio.ndim == 1, "Audio must be 1D mono."
    
    if track_id:
        cached = get_cached_top_regions(track_id, region_duration, top_k)
        if cached:
            return cached

    hop_length = 512
    
    # Compute global features ONCE for the entire array
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)

    # Setup sliding window logic
    region_frames = int(region_duration * sr / hop_length)
    hop_frames = int(hop_duration * sr / hop_length)
    
    window = np.ones(region_frames) / region_frames

    # Smooth logic
    rms_smooth = np.convolve(rms, window, mode='valid')
    centroid_smooth = np.convolve(centroid, window, mode='valid') / sr
    rolloff_smooth = np.convolve(rolloff, window, mode='valid') / sr
    onset_smooth = np.convolve(onset_env, window, mode='valid') 

    # Composite score
    scores_array = (0.4 * rms_smooth) + (0.2 * centroid_smooth) + (0.2 * rolloff_smooth) + (0.2 * onset_smooth)

    scores = []
    for i in range(0, len(scores_array), hop_frames):
        start_sec = float(i * hop_length / sr)
        scores.append((start_sec, float(scores_array[i])))

    top_regions = sorted(scores, key=lambda x: -x[1])[:top_k]
    return top_regions