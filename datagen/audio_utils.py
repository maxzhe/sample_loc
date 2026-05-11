import os
import random
from pathlib import Path
from typing import Dict, Tuple

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa
import numpy as np
from pedalboard import Pedalboard, Reverb

from config import SAMPLE_RATE, SEGMENT_LENGTH

def load_wave(path: Path, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """DRY Refactoring: Centralized wave loading logic."""
    try:
        waveform, sr_in = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"torchaudio.load failed for {path}: {e}")
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr_in != sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr_in, new_freq=sample_rate)
    return waveform

def load_random_clip(filepath: Path, length: int, sample_rate: int) -> torch.Tensor:
    waveform = load_wave(filepath, sample_rate)
    total_samples = waveform.shape[-1]
    needed = length * sample_rate
    
    # Senior-level robustness: Pad instead of discarding
    if total_samples < needed:
        padding_needed = needed - total_samples
        waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
        total_samples = needed
        
    start = random.randint(0, total_samples - needed)
    return waveform[:, start:start + needed]

def time_stretch_librosa(waveform: torch.Tensor, stretch_factor: float) -> torch.Tensor:
    y = waveform.mean(dim=0).numpy()
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    return torch.tensor(y_stretched).unsqueeze(0)

def pitch_shift_librosa(waveform: torch.Tensor, sample_rate: int, n_steps: int) -> torch.Tensor:
    y = waveform.mean(dim=0).numpy()
    y_shifted = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=n_steps)
    return torch.tensor(y_shifted).unsqueeze(0)

def apply_transformation(waveform: torch.Tensor, sample_rate: int, transform: str = "reverse", params: dict = None) -> Tuple[torch.Tensor, dict]:
    params = params or {}
    metadata = {"type": transform}

    if transform == "reverse":
        return torch.flip(waveform, dims=[-1]), metadata

    elif transform == "reverb":
        board = Pedalboard([Reverb(room_size=0.25)])
        effected = board(waveform.numpy(), sample_rate)
        return torch.tensor(effected), metadata

    elif transform == "pitch_shift":
        n_steps = params.get("pitch_shift", random.choice([-2, -1, 1, 2]))
        metadata["pitch_shift"] = n_steps
        return pitch_shift_librosa(waveform, sample_rate, n_steps), metadata

    elif transform == "time_stretch":
        stretch_factor = params.get("time_stretch", round(random.uniform(0.8, 1.2), 3))
        metadata["time_stretch"] = stretch_factor
        return time_stretch_librosa(waveform, stretch_factor), metadata

    else:
        raise NotImplementedError(f"Transformation '{transform}' is not implemented")

def insert_snippet(mix: torch.Tensor, start_sec: int, snippet_len: int, sample_rate: int,
                   transform_type: str, params: dict = None) -> Tuple[torch.Tensor, dict, torch.Tensor]:
    start_sample = start_sec * sample_rate
    end_sample = start_sample + snippet_len * sample_rate

    snippet = mix[:, start_sample:end_sample]
    transformed, metadata = apply_transformation(snippet, sample_rate, transform_type, params)

    # Check if we need to pad the mix to fit the transformed snippet
    required_length = start_sample + transformed.shape[-1]
    if required_length > mix.shape[-1]:
        padding_needed = required_length - mix.shape[-1]
        mix = torch.nn.functional.pad(mix, (0, padding_needed))
        metadata["mix_was_padded"] = True
        metadata["padding_samples"] = padding_needed
    else:
        metadata["mix_was_padded"] = False

    blended = blend_snippet(mix, transformed, start_sample)
    return blended, metadata, transformed

def mix_stems(stems: Dict[str, Path], segment_length: int = SEGMENT_LENGTH, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    mix = None
    for path in stems.values():
        clip = load_random_clip(path, segment_length, sample_rate)
        if mix is None:
            mix = clip
        else:
            min_len = min(mix.shape[-1], clip.shape[-1])
            mix = mix[:, :min_len] + clip[:, :min_len]
    return mix / len(stems) if mix is not None else torch.zeros((1, segment_length * sample_rate))

def blend_snippet(target: torch.Tensor, snippet: torch.Tensor, start_sample: int, alpha: float = 0.5) -> torch.Tensor:
    end_sample = start_sample + snippet.shape[-1]
    if end_sample > target.shape[-1]:
        raise ValueError("Snippet does not fit into target at given position")
    blended = target.clone()
    blended[:, start_sample:end_sample] = (
        (1 - alpha) * blended[:, start_sample:end_sample] + alpha * snippet
    )
    return blended

def save_wav(path: Path, waveform: torch.Tensor, sample_rate: int):
    torchaudio.save(str(path), waveform.cpu(), sample_rate)

def yamnet_filter(waveform: torch.Tensor, yamnet_model) -> bool:
    mono = waveform.mean(dim=0).numpy()
    is_music = yamnet_model.is_music(mono, SAMPLE_RATE)
    return is_music

def get_snippet_range_sec(segment_len: int, snippet_len: int) -> Tuple[int, int]:
    snippet_start = random.uniform(0, segment_len - snippet_len)
    return snippet_start, snippet_start + snippet_len