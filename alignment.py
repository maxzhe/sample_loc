import numpy as np
import random
from dataclasses import dataclass, field
from typing import Tuple, List

# =========================================================================
# The DJ Booth
# We treat audio files like songs. These tools analyze the "rhythm" and 
# figure out how to match tempos so the target aligns musically with the background.
# =========================================================================

class MusicalAnalyzer:
    def __init__(self, sr: int = 16000):
        self.sr = sr
        
    def analyze_rhythm_stability(self, beats: np.ndarray) -> str:
        """
        Looks at an array of beat timestamps [0.5, 1.0, 1.5, 2.0...] and figures out 
        if the audio has a steady metronome or if it's completely erratic (Jittery).
        If it's jittery, we shouldn't attempt smart tempo stretching, as we'll just break it.
        """
        if len(beats) < 3: return "Irregular"
        
        # Calculate the inter-beat-intervals (IBIs). The gaps between beats.
        ibis = np.diff(beats)
        med = np.median(ibis)
        
        # Filter out weird outlier gaps (like a dropped beat or a crazy drum fill)
        valid_ibis = ibis[(ibis > med * 0.5) & (ibis < med * 1.5)]
        if len(valid_ibis) < 2: return "Irregular"
        
        # Standard deviation of the gaps tells us how "tight" the rhythm is.
        std_dev = np.std(valid_ibis)
        if std_dev < 0.02: return "Locked"  # Perfect computer metronome
        if std_dev < 0.05: return "Stable"  # Good human drummer
        return "Jittery"                    # Experimental jazz

@dataclass
class MixMetadata:
    """A struct to hold our musical traits for a given file."""
    bg_key: int = 0
    bg_beats: np.ndarray = field(default_factory=lambda: np.array([]))
    tgt_key: int = 0
    tgt_beats: np.ndarray = field(default_factory=lambda: np.array([]))
    bg_stability: str = "Unknown"

def get_valid_starts(beats: np.ndarray, chunk_beats: int, ctx_sec: float) -> List[int]:
    """
    Returns a list of valid starting indices to drop our target clip into the background.
    
    Why: We don't want to drop a target right at the absolute beginning of the background 
    if the background has a 3-second silent intro. This checks the beats to find the "meat" of the song.
    """
    total = len(beats)
    if total < chunk_beats: return [0]
    
    valid_starts = list(range(0, total - chunk_beats + 1))
    if total >= 2:
        avg_beat_dist = float(np.mean(np.diff(beats)))
        
        # If the first beat doesn't happen until way late in the file, don't start at index 0.
        if float(beats[0]) > (2.0 * avg_beat_dist):
            if 0 in valid_starts and len(valid_starts) > 1:
                valid_starts.remove(0)
                
        # Don't pick an index that would run off the end of the 15-second canvas.
        last_start_idx = total - chunk_beats
        if (ctx_sec - float(beats[-1])) > (2.0 * avg_beat_dist):
            if last_start_idx in valid_starts and len(valid_starts) > 1:
                valid_starts.remove(last_start_idx)
                
    return valid_starts

def find_smart_alignment(bg_beats: np.ndarray, tgt_beats: np.ndarray, rng: random.Random, ctx_sec: float) -> Tuple[int, float]:
    """
    The actual DJ algorithm. We look at the gaps between the background beats and the 
    target beats, and find the spot where they match up with the least amount of stretching.
    
    Returns: The optimal starting index, and the mathematical 'cost' of the stretch.
    """
    if len(bg_beats) < 4 or len(tgt_beats) < 4: return 0, 999.0
    
    bg_intervals = np.diff(bg_beats)
    tgt_intervals = np.diff(tgt_beats)
    if len(bg_intervals) < len(tgt_intervals): return 0, 100.0
    
    chunk_beats = len(tgt_beats)
    valid_starts = get_valid_starts(bg_beats, chunk_beats, ctx_sec)
    if not valid_starts:
        valid_starts = [0]
    
    min_cost = float('inf')
    best_offsets = []

    # Slide the target over the background, checking every possible starting position
    for i in valid_starts:
        sub_bg = bg_intervals[i : i + len(tgt_intervals)]
        if len(sub_bg) != len(tgt_intervals): continue
        
        # Calculate how much we would need to stretch the target to hit the background marks.
        ratios = sub_bg / (tgt_intervals + 1e-9)
        # Cost is calculated logarithmically. A 1.5x stretch and a 0.66x stretch have the same 'cost'.
        cost = np.mean(np.abs(np.log2(ratios + 1e-9)))
        
        if cost < min_cost:
            min_cost = cost
            best_offsets = [i]
        elif abs(cost - min_cost) < 1e-9:
            best_offsets.append(i)

    if not best_offsets:
        return 0, 100.0

    # Prefer starting later in the file (after 2.5 seconds) if multiple offsets are tied
    valid_time_offsets = [offset for offset in best_offsets if bg_beats[offset] >= 2.5]
    if valid_time_offsets:
        chosen_offset = rng.choice(valid_time_offsets)
    else:
        chosen_offset = rng.choice(best_offsets)
        
    return chosen_offset, min_cost