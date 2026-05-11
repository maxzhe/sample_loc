import torch
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional

import torchaudio.functional as TA_F

from audio_utils import _fix_len, TARGET_SR

class TorchTimeStretcher:
    """
    Our custom, PyTorch-native Phase Vocoder.
    
    Why: Normally, changing the tempo of an audio file requires CPU-bound libraries like librosa or Rubberband. 
    That means copying tensors off the GPU to the CPU, processing, and moving them back. It's incredibly slow.
    By doing the Fast Fourier Transforms (FFT) natively in PyTorch, it runs instantly on the GPU, 
    and keeps the auto-grad graph intact in case we ever want to do end-to-end backprop.
    """
    def __init__(self, n_fft: int = 2048, win_length: int = None, hop_length: int = 512):
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.win_length)
        # Pre-compute the phase advance (the math that figures out how to shift frequencies so the pitch doesn't change when we stretch it)
        self.phase_advance = torch.linspace(0, math.pi * self.hop_length, self.n_fft // 2 + 1)[..., None]

    def _get_stft(self, x: torch.Tensor):
        if self.window.device != x.device:
            self.window = self.window.to(x.device)
            
        return torch.stft(
            x, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            center=True, 
            pad_mode='constant', # Critical: Constant padding prevents garbage noise at the absolute edges of the file.
            normalized=False, 
            return_complex=True
        )

    def _get_istft(self, x_complex: torch.Tensor, length: int = None):
        return torch.istft(
            x_complex, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            window=self.window, 
            center=True, 
            length=length
        )

    def time_stretch(self, x: torch.Tensor, rate: float) -> torch.Tensor:
        """Stretches a tensor by a constant rate. Rate 1.5 = 50% faster. Rate 0.5 = Half speed."""
        if abs(rate - 1.0) < 1e-4: return x
        
        orig_ndim = x.ndim
        if orig_ndim == 1: x = x.unsqueeze(0)
        
        spec = self._get_stft(x)
        
        if self.phase_advance.device != spec.device:
            self.phase_advance = self.phase_advance.to(spec.device)
            
        spec_stretched = TA_F.phase_vocoder(spec, rate=rate, phase_advance=self.phase_advance)
        
        expected_len = int(x.shape[-1] / rate)
        y = self._get_istft(spec_stretched, length=expected_len)
        
        if orig_ndim == 1: y = y.squeeze(0)
        return y

    def timemap_stretch(self, x: torch.Tensor, sr: int, time_map: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Dynamic, non-linear stretching based on control points.
        Imagine a DJ scratching a record—speeding up the snare fill but slowing down the bass drop.
        This function cuts the audio into chunks between mapped points and stretches each chunk independently.
        """
        if len(time_map) < 2: return x
        
        orig_ndim = x.ndim
        if orig_ndim == 1: x = x.unsqueeze(0)
        
        sorted_map = sorted(time_map, key=lambda k: k[0])
        output_segments = []
        
        for i in range(len(sorted_map) - 1):
            src_start, dst_start = sorted_map[i]
            src_end, dst_end = sorted_map[i+1]
            
            if src_end <= src_start: continue
            
            dst_dur_frames = dst_end - dst_start
            if dst_dur_frames <= 0: continue
            
            src_dur_frames = src_end - src_start
            segment_rate = src_dur_frames / dst_dur_frames
            
            chunk = x[..., src_start:src_end]
            
            # Fallback: If the chunk is so tiny it's smaller than an FFT window, we just use linear interpolation.
            # It will sound a bit weird, but phase vocoders crash on micro-chunks.
            if chunk.shape[-1] < self.n_fft:
                chunk_3d = chunk.unsqueeze(0)
                stretched_chunk = F.interpolate(chunk_3d, size=dst_dur_frames, mode='linear', align_corners=False).squeeze(0)
            else:
                stretched_chunk = self.time_stretch(chunk, rate=segment_rate)
            
            if stretched_chunk.shape[-1] != dst_dur_frames:
                stretched_chunk = _fix_len(stretched_chunk, dst_dur_frames)
                
            output_segments.append(stretched_chunk)
            
        if not output_segments:
            return torch.zeros_like(x) if orig_ndim > 1 else torch.zeros_like(x).squeeze(0)
            
        y = torch.cat(output_segments, dim=-1)
        if orig_ndim == 1: y = y.squeeze(0)
        return y

# Global Stretcher instantiation removed to avoid multiprocessing collisions.
# Now cleanly managed by the Dataset per-worker.

def apply_timemap_hq(x: torch.Tensor, time_map_sec: List[Tuple[float, float]], stretcher: Optional[TorchTimeStretcher] = None) -> Tuple[Optional[torch.Tensor], str, List[Tuple[float, float]]]:
    """Safety wrapper around the stretcher to catch corrupt time maps (like negative time, or time traveling backwards)."""
    if not time_map_sec: return x, "No map", time_map_sec
    
    if stretcher is None:
        stretcher = TorchTimeStretcher()
        
    safe_map = []
    last_t, last_b = -1.0, -1.0
    for t, b in time_map_sec:
        # Enforce strict monotonicity (time must move forward at least 1 millisecond)
        if t > last_t + 0.001 and b > last_b + 0.001:
            safe_map.append((float(t), float(b)))
            last_t, last_b = float(t), float(b)
            
    if len(safe_map) < 2:
        return None, "Invalid map (<2 points after monotonicity)", safe_map

    if safe_map[0][0] > 0.0:
        safe_map.insert(0, (0.0, 0.0))
    elif safe_map[0][0] < 0.0:
        safe_map[0] = (0.0, safe_map[0][1])

    sr = TARGET_SR
    time_map_frames = []
    for t, b in safe_map:
        time_map_frames.append((int(round(t * sr)), int(round(b * sr))))
        
    expected_samples = time_map_frames[-1][1]
    if expected_samples == 0:
        return None, "Expected samples is 0", safe_map
        
    req_src_samples = time_map_frames[-1][0]
    
    if x.shape[-1] < req_src_samples:
         x = F.pad(x, (0, req_src_samples - x.shape[-1]))
    elif x.shape[-1] > req_src_samples:
         x = x[..., :req_src_samples]

    try:
        out_tensor = stretcher.timemap_stretch(x, sr, time_map_frames)
        return out_tensor, "Success (Torch PV)", safe_map
    except Exception as e:
        return None, f"TorchStretch Error: {str(e)}", safe_map

def phase_align_stems(ref_stem: torch.Tensor, tgt_stem: torch.Tensor, sr: int = 16000, max_ms: float = 30.0) -> torch.Tensor:
    """
    Slides the target stem back and forth by a few milliseconds to ensure the audio waveforms perfectly align.
    Why: If two drum hits are mathematically identical but 5 milliseconds out of phase, they will destructively 
    interfere (cancel each other out) when mixed together, making them sound weak and hollow. 
    This uses cross-correlation to lock them into phase perfectly.
    """
    max_shift = int((max_ms / 1000.0) * sr)
    win_len = min(int(0.5 * sr), ref_stem.shape[-1], tgt_stem.shape[-1])
    if win_len < 100: return tgt_stem
        
    ref_win = ref_stem[..., :win_len].view(1, 1, -1)
    tgt_win = tgt_stem[..., :win_len].view(1, 1, -1)
    
    ref_pad = F.pad(ref_win, (max_shift, max_shift))
    corr = F.conv1d(ref_pad, tgt_win)
    best_idx = torch.argmax(corr).item()
    
    shift = best_idx - max_shift
    if shift == 0:
        return tgt_stem
    elif shift > 0:
        return F.pad(tgt_stem, (shift, 0))[..., :tgt_stem.shape[-1]]
    else:
        return F.pad(tgt_stem[..., -shift:], (0, -shift))

def apply_spectral_ducking(bg: torch.Tensor, fg_sample: torch.Tensor, sr: int = 16000, n_fft: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    Surgical Sidechain Compression.
    
    Why: If we mix a loud snare drum over a loud rock band, it will clip the audio. Standard 
    ducking just turns the whole band's volume down (which sounds weird and 'pumping'). 
    
    How: This converts the audio to frequencies (FFT). If the target snare hits at 1000Hz, we 
    carve a tiny volume hole in the background AT 1000Hz. The target slots perfectly into the background
    without altering the overall volume. This is why our mixes sound incredibly realistic.
    """
    min_len = min(bg.shape[-1], fg_sample.shape[-1])
    if min_len < n_fft: return bg[..., :min_len]
        
    bg_cut = bg[..., :min_len]
    fg_cut = fg_sample[..., :min_len]
    
    bg_2d = bg_cut.unsqueeze(0) if bg_cut.ndim == 1 else bg_cut
    fg_2d = fg_cut.unsqueeze(0) if fg_cut.ndim == 1 else fg_cut
    
    window = torch.hann_window(n_fft).to(bg.device)
    
    bg_stft = torch.stft(bg_2d, n_fft, hop_length, window=window, return_complex=True)
    fg_stft = torch.stft(fg_2d, n_fft, hop_length, window=window, return_complex=True)
    
    fg_mag = torch.abs(fg_stft)
    fg_peak = fg_mag.max()
    
    if fg_peak > 1e-6:
        duck_depth = 0.85
        fg_norm = fg_mag / fg_peak
        mask = 1.0 - (fg_norm * duck_depth)
        mask = torch.clamp(mask, min=0.15, max=1.0) # Never turn the background down past 15%
    else:
        mask = torch.ones_like(fg_mag)
        
    bg_stft_ducked = bg_stft * mask
    bg_ducked = torch.istft(bg_stft_ducked, n_fft, hop_length, window=window, length=min_len)
    
    if bg_cut.ndim == 1: return bg_ducked.squeeze(0)
    return bg_ducked

def apply_ideal_ratio_mask(mix: torch.Tensor, target: torch.Tensor, n_fft: int = 2048, hop_length: int = 512) -> torch.Tensor:
    """
    Given a Full Mix and an isolated Target, mathematically subtracts the Target to recreate the Background.
    We use the Ideal Ratio Mask (IRM) instead of pure subtraction because it sounds cleaner and prevents phase cancellation artifacts.
    """
    orig_ndim = mix.ndim
    if orig_ndim == 1:
        mix = mix.unsqueeze(0)
        target = target.unsqueeze(0)
        
    window = torch.hann_window(n_fft).to(mix.device)
    mix_stft = torch.stft(mix, n_fft, hop_length, window=window, return_complex=True, pad_mode='constant')
    tgt_stft = torch.stft(target, n_fft, hop_length, window=window, return_complex=True, pad_mode='constant')
    
    mix_mag = mix_stft.abs()
    tgt_mag = tgt_stft.abs()
    
    # Estimate the background magnitude by subtracting the target. Then apply Wiener filter logic (IRM)
    bg_mag_est = torch.relu(mix_mag - tgt_mag)
    irm_bg = (bg_mag_est ** 2) / (tgt_mag ** 2 + bg_mag_est ** 2 + 1e-8)
    
    bg_stft = mix_stft * irm_bg
    bg_time = torch.istft(bg_stft, n_fft, hop_length, window=window, length=mix.shape[-1])
    
    if orig_ndim == 1: bg_time = bg_time.squeeze(0)
    return bg_time