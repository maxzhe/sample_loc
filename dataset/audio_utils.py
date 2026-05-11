import torch
import torch.nn.functional as F
import math
from typing import Dict, Tuple, List, Optional

import torchaudio.functional as TA_F

# Unified Sample Rate Definition. 
# ASSUMPTION: The entire neural network pipeline expects 16kHz audio. 
# If we feed it 44.1kHz, the temporal convolutions will look at the wrong window sizes.
TARGET_SR = 16000

def resample_stems_to_target(src_stems: Dict[str, torch.Tensor], tgt_stems: Dict[str, torch.Tensor], orig_sr: int = 16000, target_sr: int = 16000) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Takes our dictionaries of audio stems (drums, bass, vocals, etc.) and ensures they 
    all match our TARGET_SR. 
    
    Why: We often get raw datasets at 44.1kHz or 48kHz. If we don't normalize the sample rate 
    right out of the gate, all our beat-matching math down the line will break.
    """
    resampled_src = {}
    for stem_name, audio_tensor in src_stems.items():
        if not audio_tensor.is_floating_point(): audio_tensor = audio_tensor.float()
        if orig_sr != target_sr:
            resampled_src[stem_name] = TA_F.resample(audio_tensor, orig_sr, target_sr)
        else:
            resampled_src[stem_name] = audio_tensor
        
    resampled_tgt = {}
    for stem_name, audio_tensor in tgt_stems.items():
        if not audio_tensor.is_floating_point(): audio_tensor = audio_tensor.float()
        if orig_sr != target_sr:
            resampled_tgt[stem_name] = TA_F.resample(audio_tensor, orig_sr, target_sr)
        else:
            resampled_tgt[stem_name] = audio_tensor
        
    return resampled_src, resampled_tgt

def get_active_regions(x: torch.Tensor, win_size: int = 1600, threshold_ratio: float = 0.02) -> List[int]:
    """
    Finds out where the audio is actually making noise, returning a binary list [0, 1, 1, 0...].
    
    Why: A 15-second audio file might just have a single snare hit at second 7. 
    We don't want the neural net trying to "localize" pure silence. This mask tells us exactly 
    which frames are valid for training.
    """
    if x.ndim > 1: x = x.mean(dim=0)
    peak = x.abs().max()
    # If the file is basically digital silence, return all zeros
    if peak < 1e-7: return [0] * (x.shape[-1] // win_size)

    pad_len = (win_size - (x.shape[-1] % win_size)) % win_size
    x_padded = F.pad(x, (0, pad_len))
    chunks = x_padded.view(-1, win_size)
    chunk_max = chunks.abs().max(dim=1).values
    
    # Anything above 2% of the peak volume is considered "active"
    threshold = peak * threshold_ratio
    active_mask = (chunk_max > threshold).int().tolist()
    return active_mask

def compute_smart_rms(x: torch.Tensor, frame_length: int = 2048, hop_length: int = 512, active_thresh_ratio: float = 0.05) -> float:
    """
    Measures how 'loud' the audio is, but strictly ignores the silent parts.
    
    Why: If we just took the average volume of a 15s clip with a 1s drum loop in the middle, 
    the 14 seconds of silence would artificially drag the average down. When we later try 
    to mix this to -10dB, the drum loop would get boosted to ear-bleeding levels.
    
    How: We chop the audio into tiny windows, find the loudest windows, and only calculate 
    the RMS (Root Mean Square) for those active windows.
    """
    if x.ndim > 1: x = torch.mean(x, dim=0)
    T = x.shape[-1]
    if T < frame_length: return float(torch.sqrt(torch.mean(x**2) + 1e-9))

    frames = x.unfold(-1, frame_length, hop_length)
    frame_energies = torch.mean(frames**2, dim=-1)
    frame_rms = torch.sqrt(frame_energies + 1e-9)

    peak_rms = frame_rms.max()
    if peak_rms <= 1e-9: return 0.0

    # Mask out the silence
    thresh_linear = peak_rms * active_thresh_ratio
    active_mask = frame_rms > thresh_linear
    
    if active_mask.sum() == 0: return float(peak_rms)

    active_energies = frame_energies[active_mask]
    active_rms = torch.sqrt(torch.mean(active_energies))
    return float(active_rms)

def _fix_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """
    Forces an audio tensor to be EXACTLY 'target_len' samples long.
    Neural networks hate variable-sized inputs. If it's too long, we chop the tail off. 
    If it's too short, we pad the tail with zeros (digital silence).
    """
    curr_len = x.shape[-1]
    if curr_len == target_len: return x
    elif curr_len > target_len: return x[..., :target_len]
    else: return F.pad(x, (0, target_len - curr_len))

def _fast_mono(x: torch.Tensor) -> torch.Tensor:
    """Safely crushes a stereo/multichannel tensor down to mono by averaging the channels."""
    if x.ndim == 1: return x
    if x.shape[0] == 1: return x.squeeze(0)
    return x.mean(dim=0)

def apply_crossfade(target: torch.Tensor, distractor: torch.Tensor, fade_start: int, fade_end: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a smooth DJ-style transition between the target audio and a distractor audio.
    
    Why: If we just hard-cut from one piece of audio to another, it creates an ugly "pop" 
    artifact that the neural net will memorize as a cheat code. We use a sine/cosine fade 
    to blend them smoothly over a few milliseconds so it sounds natural.
    """
    actual_fade_len = fade_end - fade_start
    if actual_fade_len > 0:
        t = torch.linspace(0, 1, actual_fade_len, device=target.device)
        gain_in, gain_out = torch.sin(t * math.pi / 2), torch.cos(t * math.pi / 2)
        if mode == "target_first":
            if fade_end < target.shape[-1]: target[..., fade_end:] = 0.0
            if fade_start > 0: distractor[..., :fade_start] = 0.0
            target[..., fade_start:fade_end] *= gain_out
            distractor[..., fade_start:fade_end] *= gain_in
        elif mode == "distractor_first":
            if fade_start > 0: target[..., :fade_start] = 0.0
            if fade_end < distractor.shape[-1]: distractor[..., fade_end:] = 0.0
            target[..., fade_start:fade_end] *= gain_in
            distractor[..., fade_start:fade_end] *= gain_out
    else:
        # Fallback to hard cut if the fade window is 0 (should rarely happen)
        if mode == "target_first":
            target[..., fade_start:] = 0.0
            distractor[..., :fade_start] = 0.0
        elif mode == "distractor_first":
            target[..., :fade_start] = 0.0
            distractor[..., fade_start:] = 0.0
    return target, distractor

def match_snr(final_target: torch.Tensor, final_bg_ducked: torch.Tensor, final_dist: Optional[torch.Tensor],
              mix_target_snr_db: Optional[float], is_passing_stem: bool, force_strict_snr: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor], str]:
    """
    The Master Volume Knob. Ensures the Target is exactly X decibels louder (or quieter) 
    than the Background.
    
    Why: Curriculum learning. We start training with the target super loud (+2dB SNR). 
    As the model gets smarter, we drop the volume into the negative decibels (-10dB) so 
    it learns to "listen" really closely to buried sounds.
    """
    action = "Unknown"
    if mix_target_snr_db is not None:
        actual_bg_rms = compute_smart_rms(final_bg_ducked)
        actual_tgt_rms = compute_smart_rms(final_target)
        
        # Don't try to divide by zero if the file is completely silent
        if actual_tgt_rms > 1e-8 and actual_bg_rms > 1e-8:
            # Physics math: Figure out exactly how loud the target needs to be based on the background
            exact_required_tgt_rms = actual_bg_rms * (10 ** (mix_target_snr_db / 20.0))
            
            # If the stem is already passing some internal threshold and we aren't forcing it:
            if is_passing_stem and not force_strict_snr:
                if actual_tgt_rms < exact_required_tgt_rms:
                    # It's too quiet, boost it to the absolute minimum floor
                    correction = exact_required_tgt_rms / actual_tgt_rms
                    final_target = final_target * correction
                    if final_dist is not None: final_dist = final_dist * correction
                    action = "Passing Stem (Boosted to Floor)"
                else:
                    action = "Passing Stem (Natural Volume Preserved)"
            else:
                # Force strictly to the requested SNR
                correction = exact_required_tgt_rms / actual_tgt_rms
                final_target = final_target * correction
                if final_dist is not None: final_dist = final_dist * correction
                action = "Strictly Aligned to Target SNR"
        else:
            action = "Skipped (Low RMS)"
    return final_target, final_dist, action