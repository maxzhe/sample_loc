from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import random
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from audio_utils import (
    TARGET_SR, _fast_mono, _fix_len, resample_stems_to_target, 
    get_active_regions, compute_smart_rms, match_snr, apply_crossfade
)
from dsp_core import (
    phase_align_stems, apply_spectral_ducking, 
    apply_ideal_ratio_mask, apply_timemap_hq, TorchTimeStretcher
)
from augmentations import apply_prefx_chain
from alignment import MusicalAnalyzer, MixMetadata, get_valid_starts, find_smart_alignment
from datatypes import AudioSample

class InfalliblePairSampler(Dataset):
    """
    The Master Assembly Line for Contrastive Audio Training.
    
    This Dataset handles the entire data curriculum: loading raw stems, analyzing beats,
    applying Digital Signal Processing (DSP) like time-stretching and EQ, mixing the stems,
    and calculating the exact 1D fractional ground-truth mapping for the neural network.
    
    By doing this heavily on CPU workers, we keep the GPUs fed and unblocked.
    """
    def __init__(
        self, root: str, output_len_sec: float = 15.0, source_context_sec: float = 15.0,
        snr_db_range: Tuple[float, float] = (-2.0, 2.0), regime: str = "C", seed: Optional[int] = None,
        prob_distractor: float = 1.0, smart_mix_prob: float = 1.0, prob_eq: float = 0.5, 
        prob_comp: float = 0.5, deterministic: bool = False, return_debug_audio: bool = False,
        enable_norm: bool = True, enable_tanh: bool = False, enable_spectral_ducking: bool = False,
        looping_prob: float = 0.50, force_strict_snr: bool = False, matrix_res: int = 46 
    ):
        self.root = Path(root)
        self.files = sorted([str(p) for p in self.root.rglob("*.pt")])
        self.regime = regime
        
        # Lengths converted to samples
        self.out_len = int(output_len_sec * TARGET_SR)
        self.ctx_len = int(source_context_sec * TARGET_SR)
        
        # Mixing curriculum probabilities
        self.snr_min, self.snr_max = snr_db_range
        self.prob_distractor = prob_distractor
        self.smart_mix_prob = smart_mix_prob
        self.prob_eq = prob_eq
        self.prob_comp = prob_comp
        self.looping_prob = looping_prob 
        self.snr_distribution = None
        self.force_strict_snr = force_strict_snr
        
        # System flags
        self.deterministic = deterministic
        self.return_debug_audio = return_debug_audio
        self.seed = seed if seed is not None else 0
        self.enable_norm = enable_norm
        self.enable_tanh = enable_tanh
        self.enable_spectral_ducking = enable_spectral_ducking
        
        # Sub-modules
        self.matrix_res = matrix_res
        self.analyzer = MusicalAnalyzer(sr=TARGET_SR)
        self.stretcher = TorchTimeStretcher()

    def __len__(self) -> int: 
        return len(self.files)

    def _load_pt(self, path: Path) -> Dict[str, Any]:
        """Safely loads the pre-processed PyTorch dictionaries."""
        return torch.load(path, map_location="cpu", weights_only=False)

    def _sum_stems(self, stem_dict: Dict[str, torch.Tensor], names: List[str]) -> torch.Tensor:
        """Helper to mix multiple tracks together into a single mono waveform."""
        out = torch.zeros(self.ctx_len, dtype=torch.float32)
        for name in names:
            if name in stem_dict:
                x = _fast_mono(stem_dict[name])
                x = _fix_len(x, self.ctx_len)
                out += x
        return out

    def __getitem__(self, idx: int) -> AudioSample:
        """
        The Main Event. PyTorch asks for an index, we return a fully mixed, 
        aligned, and mathematically mapped AudioSample struct.
        """
        # Ensure deterministic behavior for validation, but true randomness across workers for training.
        if self.deterministic:
            local_seed = int(self.seed + (idx * 99991))
            rng = random.Random(local_seed)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = worker_info.seed if worker_info is not None else self.seed
            rng = random.Random(worker_seed + idx)

        # ---------------------------------------------------------------------
        # STATION 1: Parsing & Resampling
        # ---------------------------------------------------------------------
        fpath = Path(self.files[idx])
        payload = self._load_pt(fpath)
        stems5 = payload["stems_5"]
        
        # Normalize all audio to 16kHz immediately for consistent downstream DSP
        src_all, tgt_all = resample_stems_to_target(
            stems5["source_crop"], stems5["target_crop"], orig_sr=16000, target_sr=TARGET_SR
        )
        
        schema = [str(s) for s in stems5["schema"]]
        gates = payload.get("gates", {})
        debug_info = {}
        
        # Figure out which stems are valid targets (preferably ones that passed gating)
        valid_targets = gates.get("target_db_passing_stems", [])
        if not valid_targets: valid_targets = [s for s in schema if s in tgt_all]
        if not valid_targets: valid_targets = list(tgt_all.keys())
        
        # Drums are harder to align, so we down-weight their probability of being chosen
        weighted_pool = [(s, 0.5 if "drum" in s.lower() else 1.0) for s in valid_targets]
        total_w = sum(w for _, w in weighted_pool)
        probs = [w/total_w for _, w in weighted_pool]
        stems_list = [s for s, _ in weighted_pool]
        
        # Select 1 or 2 stems to act as our 'Target'
        num_candidates = rng.choice([1, 2])
        tgt_name_list = []
        temp_stems, temp_probs = list(stems_list), list(probs)
        
        for _ in range(min(num_candidates, len(temp_stems))):
            curr_total = sum(temp_probs)
            if curr_total == 0: break
            chosen = rng.choices(temp_stems, weights=[tp/curr_total for tp in temp_probs], k=1)[0]
            tgt_name_list.append(chosen)
            
            idx_c = temp_stems.index(chosen)
            temp_stems.pop(idx_c)
            temp_probs.pop(idx_c)
             
        # Distractor Logic (adding stems that are NOT the target to confuse the model)
        bg_names = [s for s in schema if s in src_all]
        dist_components = [s for s in tgt_name_list if s in src_all]
        has_distractor = (rng.random() < self.prob_distractor and len(dist_components) > 0)
        
        if dist_components:
             bg_names = [s for s in bg_names if s not in dist_components]

        # Build Background
        bg_full = self._sum_stems(src_all, bg_names)
        bg_anchor_rms = max(compute_smart_rms(bg_full), 0.001)

        # Build Distractor
        dist_full = self._sum_stems(src_all, dist_components) if has_distractor else None
        
        # Pre-Mix the Target Stems and Balance SNR
        tgt_raw_full = torch.zeros(self.ctx_len, dtype=torch.float32)
        passing_stems = gates.get("source_db_passing_stems", [])
        
        for s_name in tgt_name_list:
            if s_name not in tgt_all: continue
            raw_stem = _fix_len(_fast_mono(tgt_all[s_name]), self.ctx_len)
            
            # Phase-align drums to prevent weird comb-filtering artifacts
            if "drum" in s_name.lower() and s_name in src_all:
                src_drum_ref = _fix_len(_fast_mono(src_all[s_name]), self.ctx_len)
                raw_stem = phase_align_stems(src_drum_ref, raw_stem, sr=TARGET_SR, max_ms=30.0)

            stem_rms = compute_smart_rms(raw_stem)
            snr_val = rng.uniform(self.snr_min, self.snr_max)
            
            # Apply curriculum SNR distributions if defined
            if self.snr_distribution:
                r_snr = rng.random()
                cum_prob = 0.0
                for snr_d in self.snr_distribution:
                    cum_prob += snr_d['prob']
                    if r_snr <= cum_prob:
                        snr_val = rng.uniform(snr_d['range'][0], snr_d['range'][1])
                        break
            
            debug_info["targeted_snr_db"] = snr_val 
            
            # Scale target volume to match desired SNR relative to background
            desired_rms = 0.0
            if stem_rms > 1e-9:
                if s_name in passing_stems and s_name in src_all and not self.force_strict_snr:
                    src_rms = compute_smart_rms(_fix_len(_fast_mono(src_all[s_name]), self.ctx_len))
                    relative_floor = bg_anchor_rms * 0.125
                    desired_rms = src_rms if (src_rms > 1e-5 and src_rms > relative_floor) else bg_anchor_rms * (10 ** (snr_val / 20.0))
                else:
                    desired_rms = bg_anchor_rms * (10 ** (snr_val / 20.0))
                
                raw_stem *= (desired_rms / stem_rms)
                
            tgt_raw_full += raw_stem
        
        # Build Reference Tensors
        if 'mix_preview' in payload and 'target_crop_mix' in payload['mix_preview']:
             target_preview_stem = _fast_mono(payload['mix_preview']['target_crop_mix'])
             reference_full = _fix_len(target_preview_stem, self.ctx_len)
        else:
             reference_full = self._sum_stems(tgt_all, [s for s in schema if s in tgt_all])

        ref_target_full = self._sum_stems(tgt_all, tgt_name_list)
        ref_bg_full = apply_ideal_ratio_mask(reference_full, ref_target_full, n_fft=2048, hop_length=512)


        # ---------------------------------------------------------------------
        # STATION 2: The DJ Booth (Beat Analysis & Mix Strategy)
        # ---------------------------------------------------------------------
        # Here we decide HOW the target fits into the background.
        # We choose between:
        # 1. Scatter Looping (Repeating a target chunk periodically)
        # 2. Smart Alignment (Warping target to match background beats)
        # 3. Random placement (If beats are unreliable)
        
        meta = MixMetadata()
        if smart_meta := payload.get('smart_metadata'):
            meta.bg_beats = np.array(smart_meta.get('source', {}).get('beats', []), dtype=np.float32)
            meta.tgt_beats = np.array(smart_meta.get('target', {}).get('beats', []), dtype=np.float32)
            meta.bg_stability = self.analyzer.analyze_rhythm_stability(meta.bg_beats)

        original_meta = MixMetadata(
            bg_key=meta.bg_key, bg_beats=meta.bg_beats.copy(), 
            tgt_key=meta.tgt_key, tgt_beats=meta.tgt_beats.copy(), 
            bg_stability=meta.bg_stability
        )

        ctx_sec = self.ctx_len / TARGET_SR
        scatter_loop_success, forced_negative, tgt_offset_sec = False, False, 0.0

        # Strategy 1: Scatter Looping (Great for creating rhythmic contrastive pairs)
        if (rng.random() < self.looping_prob) and len(meta.tgt_beats) >= 5 and len(meta.bg_beats) >= 2:
            bg_median_delta = float(np.median(np.diff(original_meta.bg_beats)))
            tgt_median_delta = float(np.median(np.diff(original_meta.tgt_beats)))
            
            # Find energetic chunks in the target to loop
            pool_size = int(0.5 * TARGET_SR)
            full_chunks = F.avg_pool1d((tgt_raw_full ** 2).view(1, 1, -1), pool_size, stride=pool_size).squeeze()
            if full_chunks.dim() == 0: full_chunks = full_chunks.unsqueeze(0)
            
            if full_chunks.numel() > 1:
                q30 = torch.quantile(torch.sqrt(full_chunks), 0.30).item()
                valid_sequences = []
                
                for N in [4, 5, 6]: # Try loops of 4, 5, or 6 beats
                    for i in get_valid_starts(original_meta.tgt_beats, N, ctx_sec):
                        if i + N >= len(meta.tgt_beats): continue
                        
                        # Verify the sequence actually has audio energy
                        active_count = sum(1 for b in meta.tgt_beats[i:i+N] if torch.sqrt(torch.mean((tgt_raw_full ** 2)[max(0, int(b * TARGET_SR) - int(0.1 * TARGET_SR)) : min(tgt_raw_full.shape[-1], int(b * TARGET_SR) + int(0.4 * TARGET_SR))])).item() >= q30)
                        if (active_count / N) >= 0.75: 
                            valid_sequences.append((i, i+N, N))
                
                # Apply the Scatter Loop
                if valid_sequences:
                    start_idx, end_idx, N = rng.choice(valid_sequences)
                    start_sec, end_sec = meta.tgt_beats[start_idx], meta.tgt_beats[end_idx]
                    start_samp, end_samp = int(start_sec * TARGET_SR), int(end_sec * TARGET_SR)
                    
                    if (chunk_len := end_samp - start_samp) > 0:
                        tgt_chunk = tgt_raw_full[start_samp:end_samp].clone()
                        rate = tgt_median_delta / bg_median_delta if bg_median_delta > 0 else 1.0
                        
                        # Time stretch to match tempo
                        try: tgt_chunk = self.stretcher.time_stretch(tgt_chunk, rate) if 0.7 <= rate <= 1.5 else tgt_chunk
                        except Exception: rate = 1.0
                            
                        # Snap to exact mathematical grid
                        if (perfect_grid_len := int(round(N * bg_median_delta * TARGET_SR))) > 0:
                            tgt_chunk = _fix_len(tgt_chunk, perfect_grid_len)
                            
                        warped_chunk_len = tgt_chunk.shape[-1]
                        fade_len = min(int(0.01 * TARGET_SR), warped_chunk_len // 2)
                        
                        # Smooth edges to prevent clicks
                        if fade_len > 0:
                            tgt_chunk[:fade_len] *= torch.linspace(0, 1, fade_len, device=tgt_chunk.device)
                            tgt_chunk[-fade_len:] *= torch.linspace(1, 0, fade_len, device=tgt_chunk.device)
                            
                        # Repeat to fill context window
                        tgt_repeated = tgt_chunk.repeat(math.ceil(self.ctx_len / warped_chunk_len) + 1)
                        
                        # Align loop start with a background beat
                        valid_bg_starts = get_valid_starts(original_meta.bg_beats, 1, ctx_sec)
                        bg_anchor_sec = float(original_meta.bg_beats[valid_bg_starts[0]]) if valid_bg_starts else 0.0
                        loop_offset_samp = (warped_chunk_len - (int(bg_anchor_sec * TARGET_SR) % warped_chunk_len)) % warped_chunk_len
                        
                        tgt_raw_full = tgt_repeated[loop_offset_samp : loop_offset_samp + self.ctx_len]
                        
                        debug_info.update({
                            "scatter_orig_start_samp": start_samp, "scatter_warped_chunk_len": warped_chunk_len,
                            "scatter_loop_offset_samp": loop_offset_samp, "scatter_rate": rate
                        })
                        scatter_loop_success = True

        # Strategy 2/3: Smart Alignment or Random Placement
        if not scatter_loop_success:
            if len(meta.tgt_beats) < 5:
                # Force negative sample if too few beats to align
                forced_negative, tgt_raw_full = True, tgt_raw_full * 0.0
            else:
                tgt_total, bg_total = len(original_meta.tgt_beats), len(original_meta.bg_beats)
                
                if tgt_total < bg_total:
                    chunk_beats = max(min(tgt_total, bg_total), rng.randint(max(1, int(tgt_total * 0.25)), max(1, int(tgt_total * 0.50)))) 
                elif tgt_total == bg_total:
                    chunk_beats = rng.randint(max(1, int(tgt_total * 0.25)), max(1, int(tgt_total * 0.50)))
                else:
                    chunk_beats = min(min(tgt_total, bg_total), rng.randint(max(1, int(bg_total * 0.25)), max(1, int(bg_total * 0.50)))) 
                    
                chunk_beats = max(1, chunk_beats)
                tgt_start_idx = rng.choice(get_valid_starts(original_meta.tgt_beats, chunk_beats, ctx_sec))
                tgt_offset_sec = float(original_meta.tgt_beats[tgt_start_idx])
                
                if (tgt_start_samp := int(tgt_offset_sec * TARGET_SR)) > 0:
                    tgt_raw_full = F.pad(tgt_raw_full[tgt_start_samp:], (0, tgt_start_samp))

        # Build Time Map for complex continuous warping
        time_map, is_smart = [], False
        if not forced_negative and not scatter_loop_success and len(meta.bg_beats) >= 4 and len(meta.tgt_beats) >= 4:
             best_offset, _ = find_smart_alignment(meta.bg_beats, meta.tgt_beats, rng, ctx_sec)
             if best_offset < len(meta.bg_beats) and meta.bg_beats[best_offset] >= 10.0:
                 early_beats = np.where(meta.bg_beats < 10.0)[0]
                 best_offset = int(early_beats[-1]) if len(early_beats) > 0 else 0
                 
             num_pts = min(len(meta.tgt_beats), len(meta.bg_beats) - best_offset)
             if meta.bg_stability != "Jittery" and num_pts > 2:
                 raw_points = [(float(meta.tgt_beats[i]), float(meta.bg_beats[best_offset + i])) for i in range(num_pts)]
                 
                 clean_map, last_tgt, last_bg = [], -1.0, -1.0
                 for t_tgt, t_bg in raw_points:
                     if t_tgt > last_tgt + 0.005 and t_bg > last_bg + 0.005:
                         clean_map.append((t_tgt, t_bg))
                         last_tgt, last_bg = t_tgt, t_bg
                 
                 if len(clean_map) >= 2:
                     t1, b1 = clean_map[0]
                     t2, b2 = clean_map[-1]
                     tgt_dur, bg_dur = t2 - t1, b2 - b1
                     slope = bg_dur / tgt_dur if tgt_dur > 0 else 1.0
                     
                     if 0.7 <= slope <= 1.5:
                         bg_start_anchor = b1 - (t1 * slope)
                         final_clean = [(float(t), float(b - bg_start_anchor)) for t, b in clean_map]
                         
                         out_sec = self.out_len / TARGET_SR
                         t1_f, b1_f = final_clean[0]
                         t2_f, b2_f = final_clean[-1]
                         
                         # Extrapolate edges to cover full context
                         t0, b0 = (0.0, max(0.0, b1_f - (t1_f * slope))) if (t1_f - (b1_f / slope)) < 0.0 else (t1_f - (b1_f / slope), 0.0)
                         t15, b15 = (out_sec, b2_f + ((out_sec - t2_f) * slope)) if (t2_f + ((out_sec - b2_f) / slope)) > out_sec else (t2_f + ((out_sec - b2_f) / slope), out_sec)
                         
                         final_clean.insert(0, (float(t0), float(b0)))
                         final_clean.append((float(t15), float(b15)))
                         time_map, is_smart = final_clean, True
                         debug_info["stretch_bg_dur"] = float(bg_dur)
        
        # ---------------------------------------------------------------------
        # STATION 3: FX Pedalboard & Audio Warping
        # ---------------------------------------------------------------------
        donor_proc = tgt_raw_full
        
        # Apply slight EQ/Comp to build robustness
        donor_proc = apply_prefx_chain(donor_proc, self.prob_eq, self.prob_comp, None, rng)
        
        # Actually perform the time stretch if we generated a Time Map
        if is_smart:
            warp_res, stretch_msg, updated_map = apply_timemap_hq(donor_proc, time_map, stretcher=self.stretcher)
            if warp_res is not None: 
                donor_proc, time_map = warp_res, updated_map
            else:
                is_smart, time_map = False, None

        # ---------------------------------------------------------------------
        # STATION 4: The Final Mix Assembly
        # ---------------------------------------------------------------------
        win_len_samp = self.out_len
        out_sec = self.out_len / TARGET_SR
        win_start, win_end = 0, win_len_samp
        
        if forced_negative or scatter_loop_success:
            tgt_dur_sec = 0.0 if forced_negative else out_sec
            distractor_mode = "none"
            handover_time = -1.0
            tgt_start_in_mix_sec = 0.0
            has_distractor = False
        else:
            tgt_dur_sec = debug_info.get("stretch_bg_dur", out_sec) if is_smart else (float(meta.tgt_beats[-1] - meta.tgt_beats[0]) if len(meta.tgt_beats) > 1 else 1.0)
            if tgt_dur_sec > out_sec: tgt_dur_sec = out_sec
            
            # Setup crossfades if a distractor is present
            if dist_full is not None and tgt_dur_sec < (out_sec - 0.5):
                has_distractor = True 
                if rng.random() < 0.5:
                    distractor_mode = "target_first"
                    tgt_start_in_mix_sec = 0.0
                    handover_time = tgt_dur_sec 
                else:
                    distractor_mode = "distractor_first"
                    tgt_start_in_mix_sec = out_sec - tgt_dur_sec
                    handover_time = tgt_start_in_mix_sec 
            else:
                has_distractor = False
                distractor_mode = "none"
                handover_time = -1.0
                tgt_start_in_mix_sec = 0.0

        if (pad_samp := int(tgt_start_in_mix_sec * TARGET_SR)) > 0:
            donor_proc = F.pad(donor_proc, (pad_samp, 0))
            if time_map is not None: time_map = [(t, b + tgt_start_in_mix_sec) for t, b in time_map]

        donor_proc = _fix_len(donor_proc, self.ctx_len)

        final_bg = _fix_len(bg_full[..., win_start:win_end], win_len_samp)
        final_target = _fix_len(donor_proc[..., win_start:win_end], win_len_samp)
        final_dist = _fix_len(dist_full[..., win_start:win_end], win_len_samp) if dist_full is not None else None

        # Execute the Handover Crossfade between Target and Distractor
        rel_handover_samp = int(handover_time * TARGET_SR) - win_start
        if has_distractor and final_dist is not None and 0 < rel_handover_samp < win_len_samp:
            fade_len = int(0.5 * TARGET_SR)
            fade_start = max(0, rel_handover_samp - (fade_len // 2))
            fade_end = min(win_len_samp, rel_handover_samp + (fade_len // 2))
            final_target, final_dist = apply_crossfade(final_target, final_dist, fade_start, fade_end, distractor_mode)
        elif final_dist is not None:
             if distractor_mode == "none" or (distractor_mode == "target_first" and rel_handover_samp >= win_len_samp) or (distractor_mode == "distractor_first" and rel_handover_samp <= 0):
                 final_dist[...] = 0.0
            
        fg_mix_pre = final_target.clone()
        if has_distractor and final_dist is not None: fg_mix_pre += final_dist
        
        # Apply Ducking and Final Mix Volume Enforcement
        final_bg_ducked = apply_spectral_ducking(final_bg, fg_mix_pre, TARGET_SR) if self.enable_spectral_ducking else final_bg.clone()
        is_passing_stem = any(s in passing_stems for s in tgt_name_list)
        
        final_target, final_dist, _ = match_snr(
            final_target, final_bg_ducked, final_dist, 
            debug_info.get("targeted_snr_db"), is_passing_stem, self.force_strict_snr
        )

        fg_mix = final_target.clone()
        if has_distractor and final_dist is not None: fg_mix += final_dist
        mixture = final_bg_ducked + fg_mix
        
        # Clip Protection
        if self.enable_norm:
            global_scale = 1.0 if (peak_amp := mixture.abs().max().item()) <= 0.99 else 0.99 / (peak_amp + 1e-7)
            mixture *= global_scale
            final_target *= global_scale
            final_bg_ducked *= global_scale
        
        if self.enable_tanh:
            mixture, final_target, final_bg_ducked = torch.tanh(mixture), torch.tanh(final_target), torch.tanh(final_bg_ducked)
        
        ref_start_sec = 0.0
        ref_start_samp = int(ref_start_sec * TARGET_SR)
        
        final_ref_crop = _fix_len(reference_full[ref_start_samp : ref_start_samp + self.out_len], self.out_len)
        final_ref_target = _fix_len(ref_target_full[ref_start_samp : ref_start_samp + self.out_len], self.out_len)
        final_ref_bg = _fix_len(ref_bg_full[ref_start_samp : ref_start_samp + self.out_len], self.out_len)
        
        if (ref_peak := final_ref_crop.abs().max()) > 0.99: 
            scale = 0.99 / (ref_peak + 1e-7)
            final_ref_crop *= scale
            final_ref_target *= scale 
            final_ref_bg *= scale 

        # ---------------------------------------------------------------------
        # STATION 5: 1D Ground Truth Tensor Generation
        # ---------------------------------------------------------------------
        # This completely unblocks the GPU by resolving the math into simple 1D grids here.
        # Output: gt_presence [Time], gt_coords [Time]
        
        sec_per_px = ctx_sec / self.matrix_res
        m_sec_1d = torch.arange(self.matrix_res, dtype=torch.float32) * sec_per_px
        
        expected_r_sec = torch.zeros(self.matrix_res, dtype=torch.float32)
        presence_1d = torch.zeros(self.matrix_res, dtype=torch.float32)
        valid_time = torch.zeros(self.matrix_res, dtype=torch.bool)
        
        is_loop_tensor = torch.tensor(scatter_loop_success, dtype=torch.bool)
        
        warped_chunk_len = debug_info.get("scatter_warped_chunk_len", 0)
        c_frames = (warped_chunk_len / TARGET_SR) / sec_per_px if (scatter_loop_success and warped_chunk_len > 0) else 1000.0 
        chunk_frames_tensor = torch.tensor(c_frames, dtype=torch.float32)
        
        # Verify physical energy exists at the coordinates
        tgt_active_list = get_active_regions(final_target, win_size=1600, threshold_ratio=0.02)
        ref_active_list = get_active_regions(final_ref_crop, win_size=1600, threshold_ratio=0.02)
        tgt_active_lowres = torch.tensor(tgt_active_list, dtype=torch.float32)
        ref_active_lowres = torch.tensor(ref_active_list, dtype=torch.float32)
        
        tgt_active = F.interpolate(tgt_active_lowres.view(1, 1, -1), size=self.matrix_res, mode='nearest').squeeze(0).squeeze(0)
        ref_active = F.interpolate(ref_active_lowres.view(1, 1, -1), size=self.matrix_res, mode='nearest').squeeze(0).squeeze(0)
        
        orig_start = debug_info.get("scatter_orig_start_samp", 0)
        loop_offset = debug_info.get("scatter_loop_offset_samp", 0)
        loop_rate = debug_info.get("scatter_rate", 1.0)
        
        # Calculate reverse mapping logic
        if time_map and not forced_negative:
            tm_bg = [x[1] for x in time_map]
            tm_tgt = [x[0] + tgt_offset_sec - ref_start_sec for x in time_map]
        else:
            tm_bg, tm_tgt = None, None
            
        if not forced_negative:
            if scatter_loop_success and warped_chunk_len > 0:
                # Resolve modulo math for looped periodic distances
                orig_start_sec = orig_start / TARGET_SR
                loop_offset_sec = loop_offset / TARGET_SR
                warped_chunk_sec = warped_chunk_len / TARGET_SR

                phase_in_warped = (m_sec_1d + loop_offset_sec) % warped_chunk_sec
                time_in_orig = phase_in_warped * loop_rate
                
                expected_r_sec = (orig_start_sec + time_in_orig) - ref_start_sec
                valid_time[:] = True 
            else:
                if time_map:
                    # Resolve piece-wise linear interpolation for Smart Alignment
                    tm_bg_t = torch.tensor(tm_bg, dtype=torch.float32)
                    tm_tgt_t = torch.tensor(tm_tgt, dtype=torch.float32)
                    
                    indices = torch.bucketize(m_sec_1d, tm_bg_t)
                    indices = torch.clamp(indices, 1, len(tm_bg_t) - 1)
                    
                    x0, x1 = tm_bg_t[indices - 1], tm_bg_t[indices]
                    y0, y1 = tm_tgt_t[indices - 1], tm_tgt_t[indices]
                    
                    weight = (m_sec_1d - x0) / (x1 - x0 + 1e-9)
                    expected_r_sec = y0 + weight * (y1 - y0)
                    valid_time[:] = True 
                else:
                    # Resolve standard linear offset
                    expected_r_sec = (m_sec_1d - tgt_start_in_mix_sec) + tgt_offset_sec - ref_start_sec
                    valid_time = (m_sec_1d >= tgt_start_in_mix_sec)
                    
            # Mask out invalid frames that fell off the edge of the matrix
            exact_r_idx_float = expected_r_sec / sec_per_px
            valid_time_mask = (exact_r_idx_float >= 0.0) & (exact_r_idx_float < self.matrix_res)
            
            r_idx_int = torch.round(exact_r_idx_float).long()
            strict_boundary_mask = valid_time_mask & (r_idx_int >= 0) & (r_idx_int < self.matrix_res)
            
            safe_r_idx_int = torch.where(strict_boundary_mask, r_idx_int, torch.zeros_like(r_idx_int))
            
            # Require both the Source AND Reference frame to have audio energy
            is_ref_active = ref_active[safe_r_idx_int] > 0.5
            is_tgt_active = tgt_active > 0.5
            
            final_presence = strict_boundary_mask & valid_time & is_tgt_active & is_ref_active
            presence_1d = final_presence.float()
        
        if forced_negative:
            presence_1d[:] = 0.0
            expected_r_sec[:] = 0.0
            
        gt_coords = expected_r_sec * presence_1d

        # Add noise floor to exactly silent mixture to avoid divide-by-zero
        if mixture.abs().max() < 1e-5: mixture += torch.randn_like(mixture) * 1e-6

        # Fulfills the strict contract defined in datatypes.py
        return AudioSample(
            mixture=mixture,
            reference=final_ref_crop,
            ref_target=final_ref_target,
            ref_bg=final_ref_bg,
            bg_ducked=final_bg_ducked,
            target_only=final_target,
            gt_presence=presence_1d,
            gt_coords=gt_coords,
            is_loop=is_loop_tensor,
            chunk_frames=chunk_frames_tensor
        )