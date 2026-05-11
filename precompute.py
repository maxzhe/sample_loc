import json
import torch
import torchaudio
import librosa
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import STEMS_ROOT, CACHE_DIR, CSV_PATH, SAMPLE_RATE
from feature_utils import find_top_audio_regions

# Required minimum stems
MIN_STEMS_PER_TRACK = 2

def estimate_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vals = np.sum(chroma, axis=1)
    pitch_class = np.argmax(chroma_vals)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[pitch_class]

def load_and_mix_stems(stem_paths):
    tensors = []
    for path in stem_paths:
        try:
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SAMPLE_RATE)
            tensors.append(wav)
        except Exception as e:
            continue
            
    if not tensors:
        return None
        
    min_len = min(t.shape[-1] for t in tensors)
    tensors = [t[:, :min_len] for t in tensors]
    mix = torch.stack(tensors, dim=0).sum(dim=0).squeeze(0)
    return mix.numpy().astype(np.float32)

def index_stems():
    stem_db = {}
    for path in STEMS_ROOT.rglob("*.*"):
        if path.suffix.lower() not in ['.wav', '.mp3']: continue
        name = path.name
        if "_(" in name and ")" in name:
            track_id = name.split("_(")[0]
            stem_db.setdefault(track_id, []).append(path)
    return {tid: paths for tid, paths in stem_db.items() if len(paths) >= MIN_STEMS_PER_TRACK}

def run_precomputation():
    print(f"🔍 Scanning {STEMS_ROOT} for audio files...")
    tracks = index_stems()
    print(f"✅ Found {len(tracks)} valid tracks.")

    if CSV_PATH.exists():
        metadata_df = pd.read_csv(CSV_PATH)
        existing_csv_ids = set(metadata_df['track_id'].astype(str))
    else:
        existing_csv_ids = set()

    print("🚀 Starting Precomputation for JSON Cache & CSV Metadata...")
    for track_id, stem_paths in tqdm(tracks.items()):
        json_path = CACHE_DIR / f"{track_id}.json"
        
        if json_path.exists() and track_id in existing_csv_ids:
            continue
            
        mix_audio = load_and_mix_stems(stem_paths)
        if mix_audio is None:
            continue

        analysis_audio = mix_audio[:SAMPLE_RATE * 60] 
        tempo_bpm, _ = librosa.beat.beat_track(y=analysis_audio, sr=SAMPLE_RATE)
        tempo_bpm = float(tempo_bpm[0]) if isinstance(tempo_bpm, np.ndarray) else float(tempo_bpm)
        bpm_group = int(round(tempo_bpm / 5.0) * 5.0)
        key_est = estimate_key(analysis_audio, SAMPLE_RATE)

        # Vectorized feature computation
        regions_5s = find_top_audio_regions(mix_audio, sr=SAMPLE_RATE, region_duration=5.0, top_k=50)
        regions_15s = find_top_audio_regions(mix_audio, sr=SAMPLE_RATE, region_duration=15.0, top_k=50)

        cache_data = {
            "bpm": tempo_bpm,
            "key": key_est,
            "5": regions_5s,
            "15": regions_15s
        }
        with open(json_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Incremental CSV saving (prevents data loss if script crashes)
        if track_id not in existing_csv_ids:
            new_row = pd.DataFrame([{
                'track_id': track_id,
                'tempo_bpm': tempo_bpm,
                'bpm_group': bpm_group,
                'key_estimate': key_est
            }])
            new_row.to_csv(CSV_PATH, mode='a', header=not CSV_PATH.exists(), index=False)
            existing_csv_ids.add(track_id)

    print("🎉 Precomputation Complete! Ready to run the Sampler.")

if __name__ == "__main__":
    run_precomputation()