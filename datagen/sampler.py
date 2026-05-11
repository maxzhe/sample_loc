# -*- coding: utf-8 -*-
"""
=============================================================================
AUDIO STEM SAMPLER & PITCH-ALIGNED PAIR GENERATOR
=============================================================================
This module creates training pairs (Source & Target) for our dataset.

DSP Lifecycle per Sample:
1. Pick a Source track and identify an active 15s region.
2. Pick a harmonically compatible Target track (using Camelot wheel metadata).
3. Extract beats for BOTH tracks using Librosa (saved to `smart_metadata`).
4. Save the ORIGINAL, un-altered Target mix to `mix_preview` for reference.
5. Calculate the exact semitone difference between Source and Target.
6. Apply Pitch Shifting to ALL Target stems so they exactly match the Source 
   track's root key (offline harmonic alignment).
7. Package everything into a `.pt` payload for `dataset.py` to ingest.
=============================================================================
"""

import os, re, random, time, traceback, zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch
import torchaudio
import librosa
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import OUTPUT_DIR, SNIPPET_LENGTH, STEMS_ROOT, CSV_PATH, BATCH_SIZE
from feature_utils import find_top_audio_regions
from audio_utils import save_wav

# ------------- Debug & constants -------------
DEBUG = True
TIMING = True
SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_FFT = 1024
DEVICE = torch.device("cpu")
MIN_STEMS_PER_TRACK = 2

CANONICAL_STEMS = ["Harmony", "Piano", "Drums", "Guitar", "Vocals"]

# Fixed crop length for both source/target (seconds)
CROP_SECONDS = 15
CROP_SAMPLES = CROP_SECONDS * SAMPLE_RATE

# Random selection defaults
PICK_STRATEGY = "weighted"   # "first" | "random_ok" | "weighted"
PICK_TEMP = 0.6              # lower -> greedier RMS preference
MAX_CANDIDATES_EVAL = 250    # cap to keep it fast
SHUFFLE_CANDIDATES = True    # randomize evaluation order
ANCHOR_RANGE = (0.2, 0.8)    # where the snippet sits inside the 15s crop (0=left edge, 1=right edge)
ANCHOR_JITTER_SEC = 0.0      # tiny extra jitter for the crop start

# =============================================
#                   Utils
# =============================================

def log(msg):
    """Robust logger that strips problematic surrogate code points."""
    s = str(msg)
    try:
        print(s, flush=True)
    except UnicodeEncodeError:
        safe = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
        try:
            print(safe, flush=True)
        except Exception:
            print("<<non-ascii log message>>", flush=True)

def tstart():
    return time.perf_counter()

def tlog(label, t0):
    if TIMING:
        dt = time.perf_counter() - t0
        log(f"⏱ {label}: {dt*1000:.1f} ms ({dt:.3f} s)")
    return time.perf_counter()

def _map_inst_to_five(name: str) -> str:
    n = name.lower()
    if "piano" in n: return "Piano"
    if "drum"  in n: return "Drums"
    if "guitar" in n: return "Guitar"
    if any(k in n for k in ["vocal","voice","singer"]): return "Vocals"
    return "Harmony"

def tensor_sum_safe(tensors):
    if not tensors: return None
    return torch.stack(tensors, dim=0).sum(dim=0)

def compute_rms(audio: torch.Tensor) -> float:
    return float((audio ** 2).mean().sqrt().item())

def windowed_rms(x: torch.Tensor, win: int, hop: int) -> torch.Tensor:
    if x.ndim == 2: x = x.squeeze(0)
    T = x.shape[-1]
    if T < win:
        return torch.tensor([x.pow(2).mean().sqrt()])
    vals = []
    for s in range(0, T - win + 1, hop):
        seg = x[s:s+win]
        vals.append(seg.pow(2).mean().sqrt())
    return torch.stack(vals)

def median_rms(audio: torch.Tensor, sr: int = SAMPLE_RATE, window_sec: float = 0.5) -> float:
    win = int(window_sec * sr); hop = win // 2
    rs = windowed_rms(audio, win, hop)
    return float(torch.median(rs).item()) if rs.numel() > 0 else compute_rms(audio)

def db_ratio(num: float, den: float, eps: float = 1e-10) -> float:
    if den <= eps:
        return float("inf") if num > eps else -float("inf")
    return 20.0 * np.log10(max(num, eps) / max(den, eps))

def stems_db_check(stem_tensors: dict, start: int, length: int):
    names = list(stem_tensors.keys())
    dB = {}
    for i, name in enumerate(names):
        s = stem_tensors[name][:, start:start+length]
        others = None
        for j, n2 in enumerate(names):
            if j == i: continue
            x = stem_tensors[n2][:, start:start+length]
            others = x if others is None else (others + x)
        stem_rms = compute_rms(s)
        others_rms = compute_rms(others) if others is not None else 1e-8
        dB[name] = db_ratio(stem_rms, others_rms)
    passing = sorted([k for k,v in dB.items() if v >= -20.0])
    return dB, passing

def convert_to_5stems(full_stems: dict) -> dict:
    out = {k: None for k in CANONICAL_STEMS}
    ref_len = None
    for v in full_stems.values():
        ref_len = v.shape[-1]; break
    if ref_len is None: ref_len = 0
    for name, wav in full_stems.items():
        bucket = _map_inst_to_five(name)
        if out[bucket] is None:
            out[bucket] = wav.clone()
        else:
            L = min(out[bucket].shape[-1], wav.shape[-1])
            out[bucket] = out[bucket][:, :L] + wav[:, :L]
    for k in CANONICAL_STEMS:
        if out[k] is None:
            out[k] = torch.zeros(1, ref_len)
    return out

def align_lengths_to_min(tensors: list) -> list:
    if not tensors: return []
    min_len = min(t.shape[-1] for t in tensors)
    return [t[:, :min_len] for t in tensors]

def load_wave(path):
    try:
        waveform, sr_in = torchaudio.load(str(path))
    except Exception as e:
        raise RuntimeError(f"torchaudio.load failed for {path}: {e}")
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr_in != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr_in, new_freq=SAMPLE_RATE)
    return waveform

def slice_exact_length(wav: torch.Tensor, start: int, desired_len: int) -> torch.Tensor:
    end = min(start + desired_len, wav.shape[-1])
    seg = wav[:, start:end]
    need = desired_len - seg.shape[-1]
    if need > 0:
        pad = torch.zeros((1, need), dtype=seg.dtype, device=seg.device)
        seg = torch.cat([seg, pad], dim=-1)
    elif need < 0:
        seg = seg[:, :desired_len]
    return seg

def compute_anchored_crop_start(
    snippet_start: int, snippet_len: int, track_len: int, desired_len: int,
    anchor_range=(0.2, 0.8), jitter_sec: float = 0.0
) -> int:
    lo = snippet_start + snippet_len - desired_len
    hi = snippet_start
    lo = max(0, min(lo, track_len - desired_len))
    hi = max(0, min(hi, track_len - desired_len))
    if hi < lo:
        lo = hi = max(0, min(snippet_start - desired_len // 2, track_len - desired_len))

    a0, a1 = anchor_range
    a = random.uniform(a0, a1)
    start = int(round(lo + a * (hi - lo)))

    if jitter_sec > 0:
        j = int(round(random.uniform(-jitter_sec, jitter_sec) * SAMPLE_RATE))
        start = max(0, min(track_len - desired_len, start + j))

    return int(start)

camelot_map = {
    'A':'8A','A#':'3A','B':'10A','C':'8B','C#':'12A','D':'10B','D#':'1A','E':'12B','F':'7B','F#':'2B','G':'9B','G#':'11A'
}

def get_matching_keys(source_key, mode='smooth'):
    code = camelot_map.get(source_key)
    if not code: return set()
    num, ring = int(code[:-1]), code[-1]
    smooth = {code, f"{(num % 12) + 1}{ring}", f"{((num + 10) % 12) + 1}{ring}", f"{num}{'B' if ring=='A' else 'A'}"}
    advanced = {f"{num + (1 if ring=='A' else -1)}{'B' if ring=='A' else 'A'}",
                f"{((num + 6 - 1) % 12) + 1}{ring}", f"{((num + 1 - 1) % 12) + 1}{ring}"}
    codes = smooth if mode=='smooth' else (smooth | advanced)
    rev = {v:k for k,v in camelot_map.items()}
    return {rev[c] for c in codes if c in rev}

def get_pitch_shift_steps(src_key: str, tgt_key: str) -> int:
    """
    Calculates the shortest semitone shift required to transpose tgt_key to src_key.
    Returns an int between -6 and +5.
    """
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    if not src_key or not tgt_key or src_key not in notes or tgt_key not in notes:
        return 0
    s_idx = notes.index(src_key)
    t_idx = notes.index(tgt_key)
    diff = s_idx - t_idx
    # Find shortest distance on the 12-tone circle
    shift = (diff + 6) % 12 - 6
    return shift

class Sampler:
    def __init__(self):
        if not STEMS_ROOT.exists():
            raise RuntimeError(f"STEMS_ROOT does not exist: {STEMS_ROOT}")
        ext_list = ["*.wav", "*.mp3"]
        all_files = []
        for pat in ext_list:
            all_files.extend(list(STEMS_ROOT.rglob(pat)))
        log(f"🔎 STEMS_ROOT={STEMS_ROOT} | found {len(all_files)} stem files")

        self.track_metadata = None
        try:
            self.track_metadata = pd.read_csv(CSV_PATH)
            log(f"📄 Loaded metadata: {CSV_PATH} ({len(self.track_metadata)} rows)")
        except Exception as e:
            log(f"ℹ️ No metadata or failed to read: {e}. Will fallback to random target.")

        self.target_tempo = self.source_tempo = None
        self.target_key = self.source_key = None
        self._index_stems(ext_list)

    def _index_stems(self, ext_list):
        t0 = tstart()
        self._stem_db = {}
        total = 0; matched = 0; skipped = 0

        log("📦 Indexing stems (pattern '<trackId>_(<Instrument>).ext')...")
        for pat in ext_list:
            for path in STEMS_ROOT.rglob(pat):
                total += 1
                name = path.name
                if "_(" not in name or ")" not in name:
                    skipped += 1
                    continue
                try:
                    track_id = name.split("_(")[0]
                    inst = name.split("(")[-1].split(")")[0]
                    self._stem_db.setdefault(track_id, {})[inst] = path
                    matched += 1
                except Exception:
                    skipped += 1

        self._valid_track_ids = [tid for tid, stems in self._stem_db.items() if len(stems) >= MIN_STEMS_PER_TRACK]
        log(f"✅ Indexed tracks={len(self._stem_db)} | with >= {MIN_STEMS_PER_TRACK} stems={len(self._valid_track_ids)} "
            f"| files matched={matched}/{total} | skipped={skipped}")
        tlog("index_stems", t0)

        if len(self._valid_track_ids) == 0:
            raise RuntimeError("No valid tracks found. Check your file naming and structure.")

    def _pick_full_track(self):
        if not self._valid_track_ids:
            raise RuntimeError("No valid tracks available to pick from.")
        tid = random.choice(self._valid_track_ids)
        
        # Get Source Key for reference
        df = self.track_metadata
        if df is not None and not df.empty:
            row = df[df.track_id == tid]
            self.source_key = row.iloc[0].get("key_estimate", None) if not row.empty else None
            
        return tid, self._stem_db[tid]

    def _fallback_target(self, exclude_id):
        candidates = [tid for tid in self._valid_track_ids if tid != exclude_id]
        if not candidates:
            raise RuntimeError("Need at least 2 valid tracks for target fallback.")
        tid = random.choice(candidates)
        
        # Get Target Key for fallback
        df = self.track_metadata
        if df is not None and not df.empty:
            row = df[df.track_id == tid]
            self.target_key = row.iloc[0].get("key_estimate", None) if not row.empty else None
            
        return tid, self._stem_db[tid]

    def _pick_different_track(self, source_id, match_mode='smooth'):
        df = self.track_metadata
        if df is None or df.empty:
            return self._fallback_target(source_id)
        src_meta = df[df.track_id == source_id]
        if src_meta.empty:
            return self._fallback_target(source_id)
            
        row = src_meta.iloc[0]
        self.source_tempo = row.get("tempo_bpm", None)
        self.source_key = row.get("key_estimate", None)

        valid_keys = get_matching_keys(self.source_key, mode=match_mode) if self.source_key else set()
        bpm_group = row.get("bpm_group", None)
        if bpm_group is None or not valid_keys:
            return self._fallback_target(source_id)

        cand = df[(df.bpm_group == bpm_group) & (df.track_id != source_id) & (df.key_estimate.isin(valid_keys))]
        valid_ids = [tid for tid in cand.track_id if tid in self._stem_db]
        if not valid_ids:
            return self._fallback_target(source_id)
            
        target_id = random.choice(valid_ids)
        tgt_row = cand[cand.track_id == target_id].iloc[0]
        self.target_key = tgt_row.get("key_estimate", None)
        
        return target_id, self._stem_db[target_id]

    def _choose_region_with_checks(self, mix_tensor, stem_tensors: dict, track_id: str, window_sec: float, pick_strategy: str = PICK_STRATEGY, pick_temp: float = PICK_TEMP, max_candidates_eval: int = MAX_CANDIDATES_EVAL, shuffle_candidates: bool = SHUFFLE_CANDIDATES):
        y = mix_tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)

        onset_env = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH)
        peak_idx = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=3, delta=0.0, wait=4)
        peak_idx = list(map(int, peak_idx)) if hasattr(peak_idx, "__iter__") else []
        times = librosa.frames_to_time(np.arange(len(onset_env)), sr=SAMPLE_RATE, hop_length=HOP_LENGTH)

        top_regions = find_top_audio_regions(audio=y, sr=SAMPLE_RATE, region_duration=window_sec, top_k=10, hop_duration=0.5, track_id=track_id)

        region_starts = [float(max(0.0, times[pi] - 0.03)) for pi in peak_idx if 0 <= pi < len(times)]
        if top_regions:
            region_starts.extend([float(r[0]) for r in top_regions])

        total_sec = y.shape[-1] / SAMPLE_RATE
        if total_sec > window_sec:
            grid = np.linspace(0.0, max(0.0, total_sec - window_sec), num=12, endpoint=True)
            region_starts.extend(map(float, grid))

        seen, unique_starts = set(), []
        for s in region_starts:
            k = round(s, 3)
            if k not in seen:
                unique_starts.append(s); seen.add(k)
        if shuffle_candidates:
            random.shuffle(unique_starts)

        if not unique_starts:
            hop_sec = 0.5
            unique_starts = [float(t) for t in np.arange(0, max(0.0, total_sec - window_sec), hop_sec)]
            if shuffle_candidates:
                random.shuffle(unique_starts)

        med_r = median_rms(mix_tensor, sr=SAMPLE_RATE, window_sec=0.5)
        thr = 0.3 * med_r
        frame_len = int(window_sec * SAMPLE_RATE)

        five = convert_to_5stems(stem_tensors)
        passed = []
        scanned = 0

        for start in unique_starts:
            if scanned >= max_candidates_eval: break
            scanned += 1
            s_i = int(start * SAMPLE_RATE)
            e_i = s_i + frame_len
            if e_i > mix_tensor.shape[-1]: continue

            seg = mix_tensor[:, s_i:e_i]
            seg_r = compute_rms(seg)
            if seg_r < thr: continue

            dB, passing = stems_db_check(five, s_i, frame_len)
            if len(passing) >= 3:
                passed.append((start, seg_r, dB, passing))

        if not passed: return None, None, None, None, None

        if pick_strategy == "first":
            choice = passed[0]
        elif pick_strategy == "random_ok":
            choice = random.choice(passed)
        else:
            segs = np.array([p[1] for p in passed], dtype=np.float32)
            z = (segs / max(1e-8, pick_temp))
            z = z - z.max()
            w = np.exp(z)
            w = w / w.sum()
            choice = random.choices(passed, weights=w.tolist(), k=1)[0]

        return choice[0], choice[1], med_r, choice[2], choice[3]

    def _maybe_zip_batch(self, batch_dir: Path, batch_idx: int):
        pts = list(batch_dir.glob("*.pt"))
        if len(pts) < BATCH_SIZE: return
        zip_path = OUTPUT_DIR / f"prepared_batch_{batch_idx:05d}.zip"
        if zip_path.exists(): return
        
        log(f"📦 Zipping batch {batch_idx} (files={len(pts)}) -> {zip_path.name}")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(pts):
                if p.suffix.lower() == ".pt":
                    zf.write(p, arcname=p.name)
        
        for p in sorted(pts):
            try: p.unlink()
            except Exception as e: log(f"⚠️ Could not delete {p.name}: {e}")

    def generate_sample(self, sample_idx: int):
        for attempt in range(10):
            try:
                log(f"\n🌀 Attempt {attempt+1}/10 — Prepared {sample_idx:05d}")

                # ==========================================
                # 1. SOURCE TRACK PROCESSING
                # ==========================================
                source_id, source_stem_paths = self._pick_full_track()
                source_loaded = {k: load_wave(p) for k, p in source_stem_paths.items()}
                src_tensors = align_lengths_to_min(list(source_loaded.values()))
                if not src_tensors: continue
                source_mix = tensor_sum_safe(src_tensors)
                
                src_start_sec, src_rms, src_med, src_db_map, src_pass = self._choose_region_with_checks(
                    source_mix, source_loaded, source_id, SNIPPET_LENGTH
                )
                if src_start_sec is None: continue
                
                start_s = int(src_start_sec * SAMPLE_RATE)
                L_snip = int(SNIPPET_LENGTH * SAMPLE_RATE)

                track_len = source_mix.shape[-1]
                src_crop_start = compute_anchored_crop_start(
                    start_s, L_snip, track_len, CROP_SAMPLES, anchor_range=ANCHOR_RANGE, jitter_sec=ANCHOR_JITTER_SEC
                )
                source_crop_stems = {k: slice_exact_length(v, src_crop_start, CROP_SAMPLES) for k,v in source_loaded.items()}
                source_crop_5 = convert_to_5stems(source_crop_stems)
                src_crop_mix = tensor_sum_safe(list(source_crop_5.values()))

                # ==========================================
                # 2. TARGET TRACK PROCESSING
                # ==========================================
                target_id, target_stem_paths = self._pick_different_track(source_id)
                target_loaded = {k: load_wave(p) for k, p in target_stem_paths.items()}
                tgt_tensors = align_lengths_to_min(list(target_loaded.values()))
                if not tgt_tensors: continue
                target_mix = tensor_sum_safe(tgt_tensors)

                tgt_start_sec, tgt_rms, tgt_med, tgt_db_map, tgt_pass = self._choose_region_with_checks(
                    target_mix, target_loaded, target_id, SNIPPET_LENGTH
                )
                if tgt_start_sec is None: continue
                tgt_start_s = int(tgt_start_sec * SAMPLE_RATE)

                tgt_track_len = target_mix.shape[-1]
                tgt_crop_start = compute_anchored_crop_start(
                    tgt_start_s, L_snip, tgt_track_len, CROP_SAMPLES, anchor_range=ANCHOR_RANGE, jitter_sec=ANCHOR_JITTER_SEC
                )
                target_crop_stems = {k: slice_exact_length(v, tgt_crop_start, CROP_SAMPLES) for k,v in target_loaded.items()}
                target_crop_5 = convert_to_5stems(target_crop_stems)
                
                # We need the original sum *before* any pitch shifting occurs!
                original_tgt_crop_mix = tensor_sum_safe(list(target_crop_5.values()))

                # ==========================================
                # 3. BEAT EXTRACTION (SMART METADATA)
                # ==========================================
                src_mix_np = src_crop_mix.squeeze(0).numpy()
                tgt_mix_np = original_tgt_crop_mix.squeeze(0).numpy()

                _, src_beats = librosa.beat.beat_track(y=src_mix_np, sr=SAMPLE_RATE)
                src_beats_sec = librosa.frames_to_time(src_beats, sr=SAMPLE_RATE).tolist()

                _, tgt_beats = librosa.beat.beat_track(y=tgt_mix_np, sr=SAMPLE_RATE)
                tgt_beats_sec = librosa.frames_to_time(tgt_beats, sr=SAMPLE_RATE).tolist()

                # ==========================================
                # 4. TARGET PITCH SHIFTING (CAMELOT WHEEL)
                # ==========================================
                pitch_steps = get_pitch_shift_steps(self.source_key, self.target_key)
                if pitch_steps != 0:
                    log(f"   ↳ Pitch shifting Target stems by {pitch_steps} semitones ({self.target_key} -> {self.source_key})")
                    for k, v in target_crop_5.items():
                        if v is not None and v.sum() != 0:
                            try:
                                # Apply PyTorch-native pitch shifting to all stems individually
                                target_crop_5[k] = torchaudio.functional.pitch_shift(v, SAMPLE_RATE, n_steps=pitch_steps)
                            except Exception as e:
                                log(f"⚠️ Pitch shift failed for {k}: {e}")

                # ==========================================
                # 5. PAYLOAD ASSEMBLY
                # ==========================================
                payload = {
                    "schema": "prepared_pair_v6",
                    "ids": {"source_track_id": source_id, "target_track_id": target_id, "sample_gen_id": int(sample_idx)},
                    "params": {"sample_rate": SAMPLE_RATE, "n_fft": N_FFT, "hop_length": HOP_LENGTH},
                    "stems_5": {
                        "schema": CANONICAL_STEMS,
                        "source_crop": {k: v.cpu() for k,v in source_crop_5.items()},
                        "target_crop": {k: v.cpu() for k,v in target_crop_5.items()}, # These are now pitch-shifted
                    },
                    "gates": {
                        "source_db_passing_stems": src_pass,
                        "target_db_passing_stems": tgt_pass,
                    },
                    "smart_metadata": {
                        "source": {"beats": src_beats_sec},
                        "target": {"beats": tgt_beats_sec}
                    },
                    "mix_preview": {
                        # Original target mix is saved entirely un-altered
                        "target_crop_mix": original_tgt_crop_mix.cpu()
                    }
                }

                batch_idx = sample_idx // BATCH_SIZE
                batch_dir = OUTPUT_DIR / f"prepared_batch_{batch_idx:05d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                pt_name = f"{source_id}_{target_id}_{sample_idx:05d}.pt"
                pt_path = batch_dir / pt_name
                torch.save(payload, pt_path)

                if (sample_idx % 20) == 0:
                    if src_crop_mix is not None: save_wav(batch_dir / f"{pt_name[:-3]}_src_15s.wav", src_crop_mix, SAMPLE_RATE)
                    if original_tgt_crop_mix is not None: save_wav(batch_dir / f"{pt_name[:-3]}_tgt_15s.wav", original_tgt_crop_mix, SAMPLE_RATE)

                self._maybe_zip_batch(batch_dir, batch_idx)
                log(f"✅ Saved {pt_name}")
                return pt_path

            except Exception as e:
                log(f"⚠️ Attempt {attempt+1} failed: {e}")
        return None

def _worker_run(start_idx: int, n: int):
    random.seed() 
    np.random.seed()
    torch.manual_seed(random.randint(0, 2**31 - 1))
    sampler = Sampler()
    end = start_idx + n
    for i in range(start_idx, end):
        sampler.generate_sample(i)
    return True

if __name__ == "__main__":
    existing_pts = list(OUTPUT_DIR.rglob("prepared_batch_*/*.pt"))
    existing_ids = []
    for p in existing_pts:
        m = re.match(r".*_(\d{5})\.pt$", p.name)
        if m: existing_ids.append(int(m.group(1)))
    start_idx = max(existing_ids) + 1 if existing_ids else 0

    total = 10000 
    workers = max(1, mp.cpu_count() // 2) 
    chunk = 50 

    if workers == 1:
        _worker_run(start_idx, total)
    else:
        tasks = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            i = start_idx
            end = start_idx + total
            while i < end:
                n = min(chunk, end - i)
                tasks.append(ex.submit(_worker_run, i, n))
                i += n
            for fut in as_completed(tasks):
                _ = fut.result()