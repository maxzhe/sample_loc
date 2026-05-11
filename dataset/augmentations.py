import torch
import torch.nn.functional as F
import random
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import torchaudio.functional as TA_F

from audio_utils import TARGET_SR

# =========================================================================
# The Pedalboard
# These functions modify the target audio to make it sound "grittier" or 
# distinct, preventing the network from overfitting to clean, studio-quality samples.
# =========================================================================

@dataclass
class FXEvent:
    """A log entry recording what pedal we stepped on."""
    phase: str
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: Optional[str] = None

@dataclass
class FXTrace:
    """A running history of the entire pedalboard chain for debugging."""
    pre: List[FXEvent] = field(default_factory=list)
    post: List[FXEvent] = field(default_factory=list)
    
    def record(self, phase: str, name: str, params: Dict[str, Any], ok: bool = True, error: Optional[str] = None):
        ev = FXEvent(phase=phase, name=name, params=params, ok=ok, error=error)
        (self.pre if phase == "pre" else self.post).append(ev)
        
    def as_dict(self) -> Dict[str, Any]:
        def _ev(e): return {"phase": e.phase, "name": e.name, "params": e.params, "ok": e.ok, "error": e.error}
        return {"pre": [_ev(e) for e in self.pre], "post": [_ev(e) for e in self.post]}

def fx_band_eq(x: torch.Tensor, trace: Optional[FXTrace], phase: str, rng: random.Random) -> torch.Tensor:
    """
    Randomly boosts or cuts up to 5 different frequency bands.
    Why: Real-world audio comes from different microphones, phones, and rooms. 
    By constantly shifting the EQ curve, we force the model to learn the structural 
    timbre of the sound rather than just memorizing its EQ fingerprint.
    """
    params = {}
    try:
        n_bands = rng.randint(1, 5)
        bands_info = []
        for _ in range(n_bands):
            g = rng.uniform(-10, 5)   # Gain: Cut down to -10dB, or boost up to +5dB
            f = rng.uniform(60, 8000) # Frequency: Anywhere from sub-bass to high treble
            q = rng.uniform(0.5, 1.2) # Q-factor: How wide or narrow the EQ bell curve is
            if x.ndim == 1: x = TA_F.equalizer_biquad(x.unsqueeze(0), TARGET_SR, f, g, q).squeeze(0)
            else: x = TA_F.equalizer_biquad(x, TARGET_SR, f, g, q)
            bands_info.append((f, g, q))
        params["bands"] = bands_info
        if trace: trace.record(phase, "BandEQ", params)
    except Exception as e:
        if trace: trace.record(phase, "BandEQ", params, ok=False, error=str(e))
    return x

def fx_clean_compression(x: torch.Tensor, trace: Optional[FXTrace], phase: str, rng: random.Random) -> torch.Tensor:
    """
    Dynamic Range Compression. Squashes loud peaks to make the audio thicker and punchier.
    
    Why: We use 'smart' compression based on Crest Factor. If the audio is very spiky 
    (like a drum beat - crest factor > 4.0), we attack it fast and hard. If it's smooth 
    (like a synth or vocal - crest factor < 4.0), we compress it gently.
    """
    params = {"type": "clean_comp"}
    try:
        peak = x.abs().max()
        rms = torch.sqrt(torch.mean(x**2) + 1e-8)
        cf = float(peak / rms) if rms > 1e-9 else 0.0
        
        params["crest_factor"] = cf
        
        # Spiky Audio (Drums) -> Hard limiting
        if cf > 4.0: 
            ratio = rng.uniform(1.5, 3.0)
            threshold_db = rng.uniform(-15.0, -8.0)
            attack_ms = rng.uniform(10.0, 25.0)
            makeup_db = rng.uniform(0.0, 1.5)
        # Smooth Audio (Vocals) -> Gentle squeeze
        else:
            ratio = rng.uniform(3.0, 6.0)
            threshold_db = rng.uniform(-20.0, -12.0)
            attack_ms = rng.uniform(2.0, 10.0)
            makeup_db = rng.uniform(2.0, 4.0)
            
        ndim = x.ndim
        if ndim == 1: x_proc = x.unsqueeze(0)
        else: x_proc = x
        
        # Calculate how big our envelope detector window should be
        kernel_size = int((attack_ms / 1000.0) * TARGET_SR)
        if kernel_size < 1: kernel_size = 1
        if kernel_size % 2 == 0: kernel_size += 1
        
        x_abs = x_proc.abs()
        # Smooth the audio out to trace its volume envelope over time
        envelope = F.avg_pool1d(x_abs.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze(1)
        env_db = 20 * torch.log10(envelope + 1e-6)
        
        # Math: Figure out how far over the threshold we are, calculate the gain reduction, and turn the volume down
        overshoot = torch.relu(env_db - threshold_db)
        gain_reduction_db = overshoot * (1.0 - 1.0/ratio)
        gain_linear = 10 ** (-gain_reduction_db / 20.0)
        
        y = x_proc * gain_linear * (10 ** (makeup_db / 20.0))
        if ndim == 1: y = y.squeeze(0)
        if trace: trace.record(phase, "CleanComp", params)
        return y
    except Exception as e:
        if trace: trace.record(phase, "CleanComp", params, ok=False, error=str(e))
        return x

def apply_prefx_chain(x: torch.Tensor, prob_eq: float, prob_comp: float, trace: Optional[FXTrace], rng: random.Random) -> torch.Tensor:
    """The master pedalboard toggle. Rolls the dice to decide which FX to turn on."""
    if rng.random() < prob_eq: x = fx_band_eq(x, trace, "pre", rng)
    if rng.random() < prob_comp: x = fx_clean_compression(x, trace, "pre", rng)
    return x