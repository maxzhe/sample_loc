from __future__ import annotations

import time
import os
import argparse
from typing import Tuple, List, Any

import torch
from torch.utils.data import DataLoader, Subset, Dataset
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt

# Using the refactored, cleanly modularized pipeline
from dataset import InfalliblePairSampler
from audio_utils import TARGET_SR
from contracts import AudioSample, AudioBatch

class LoopingWrapper(Dataset):
    """
    [PHASE III] Wraps the dataset to dynamically handle looping probabilities from the curriculum.
    """
    def __init__(self, base_ds: Dataset, looping_prob: float = 0.5):
        self.ds = base_ds
        self.looping_prob = looping_prob

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> AudioSample:
        # Fulfills the AudioSample contract identically
        return self.ds[idx]

def collate_audio_batch(batch: List[AudioSample]) -> AudioBatch:
    return AudioBatch(
        mixture=torch.stack([b.mixture for b in batch]),
        reference=torch.stack([b.reference for b in batch]),
        ref_target=torch.stack([b.ref_target for b in batch]),
        ref_bg=torch.stack([b.ref_bg for b in batch]),
        bg_ducked=torch.stack([b.bg_ducked for b in batch]),
        target_only=torch.stack([b.target_only for b in batch]),
        gt_presence=torch.stack([b.gt_presence for b in batch]),
        gt_coords=torch.stack([b.gt_coords for b in batch]),
        is_loop=torch.stack([b.is_loop for b in batch]),
        chunk_frames=torch.stack([b.chunk_frames for b in batch])
    )

class SampleIDDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Any):
        super().__init__()
        self.cfg = cfg
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: str | None = None):
        val_split = self.cfg.data.get('val_split', 0.1)
        looping_prob = self.cfg.data.get('looping_prob', 0.5)
        force_strict_snr = self.cfg.data.get('force_strict_snr', False)
        seed = self.cfg.data.get('seed', 0)
        matrix_res = self.cfg.data.get('matrix_res', 46)

        ds_train_base = InfalliblePairSampler(
            root=self.cfg.data.data_root, 
            source_context_sec=self.cfg.data.source_sec,
            output_len_sec=self.cfg.data.ref_sec, 
            snr_db_range=self.cfg.data.snr_db_range,
            prob_distractor=self.cfg.data.prob_distractor, 
            return_debug_audio=False,
            deterministic=False, 
            seed=seed,
            force_strict_snr=force_strict_snr,
            matrix_res=matrix_res
        )
        
        ds_val_base = InfalliblePairSampler(
            root=self.cfg.data.data_root, 
            source_context_sec=self.cfg.data.source_sec,
            output_len_sec=self.cfg.data.ref_sec, 
            snr_db_range=self.cfg.data.snr_db_range,
            prob_distractor=self.cfg.data.prob_distractor, 
            return_debug_audio=False,
            deterministic=True, 
            seed=seed,
            force_strict_snr=force_strict_snr,
            matrix_res=matrix_res
        )

        ds_train_wrapped = LoopingWrapper(ds_train_base, looping_prob=looping_prob)
        ds_val_wrapped = LoopingWrapper(ds_val_base, looping_prob=looping_prob)

        n = len(ds_train_base)
        val_size = int(n * val_split)
        n_train = n - val_size
        
        # Ensure a randomized, non-sequential split to prevent class leakage and validation bias.
        generator = np.random.RandomState(seed)
        indices = generator.permutation(n).tolist()
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        self.ds_train = Subset(ds_train_wrapped, train_indices)
        self.ds_val = Subset(ds_val_wrapped, val_indices)

    def update_curriculum(self, snr_range: Tuple[float, float], seq_len_sec: float, loop_prob: float, dist_prob: float, snr_dist=None):
        for subset in [self.ds_train, self.ds_val]:
            if subset is None: continue
            
            wrapper = subset.dataset
            wrapper.looping_prob = loop_prob
            
            base = wrapper.ds
            base.snr_min, base.snr_max = snr_range
            base.prob_distractor = dist_prob
            base.snr_distribution = snr_dist
            
            base.ctx_len = int(seq_len_sec * TARGET_SR)
            base.out_len = int(seq_len_sec * TARGET_SR)

    def train_dataloader(self):
        return DataLoader(
            self.ds_train, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=True, 
            num_workers=self.cfg.training.num_workers,
            drop_last=True,
            collate_fn=collate_audio_batch
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val, 
            batch_size=self.cfg.training.batch_size, 
            shuffle=False,
            num_workers=self.cfg.training.num_workers,
            drop_last=True,
            collate_fn=collate_audio_batch
        )

# =========================================================================================
# DIAGNOSTICS & SMOKETESTS
# =========================================================================================

def visualize_dataloader_batch(data_root: str, idx: int = 0, matrix_res: int = 46, out_path: str = "dataloader_1d_targets.png"):
    print(f"\n🚀 Running 1D Dataloader Visual Smoke Test on Index {idx}...")
    if not os.path.exists(data_root):
        print(f"[ERROR] Data root not found: {data_root}"); return

    dataset = InfalliblePairSampler(root=data_root, source_context_sec=15.0, output_len_sec=15.0, deterministic=True, matrix_res=matrix_res)
    if len(dataset) == 0: print("Dataset is empty!"); return

    actual_idx = min(idx, len(dataset) - 1)
    
    # Directly pulling the AudioSample dataclass contract
    item = dataset[actual_idx]
    batch = collate_audio_batch([item])

    presence = batch.gt_presence[0].numpy()
    coords = batch.gt_coords[0].numpy()
    
    coords_plot = np.where(presence > 0.5, coords, np.nan)
    
    ctx_sec = batch.mixture.shape[-1] / TARGET_SR
    time_axis = np.linspace(0, ctx_sec, matrix_res)

    print(f"✅ Extracted 1D Targets | Shape: {presence.shape}")
    print(f"✅ Target Active Frames: {presence.sum():.0f} / {matrix_res}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(time_axis, presence, drawstyle='steps-mid', color='red', linewidth=2)
    ax1.fill_between(time_axis, presence, step='mid', alpha=0.3, color='red')
    ax1.set_title(f"Target Presence (Index {actual_idx})")
    ax1.set_ylabel("Presence Probability")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(time_axis, coords_plot, color='blue', linewidth=3, label="Expected Coordinate (Linear)")
    ax2.scatter(time_axis[presence > 0.5], coords_plot[presence > 0.5], color='black', s=10)
    ax2.set_title("Reference Coordinate Regression")
    ax2.set_xlabel("Mixture Time (seconds)")
    ax2.set_ylabel("Reference Time (seconds)")
    ax2.set_ylim(0, ctx_sec)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    print(f"📸 Saved exact 1D Dataloader plots to: {out_path}\n")

def run_pipeline_benchmark(data_root: str, num_samples: int = 32, matrix_res: int = 46, batch_size: int = 8, num_workers: int = 8):
    print("\n" + "="*65 + "\n🚀 PIPELINE & DATALOADER BENCHMARK 🚀\n" + "="*65)
    dataset = InfalliblePairSampler(root=data_root, source_context_sec=15.0, output_len_sec=15.0, deterministic=True, matrix_res=matrix_res)
    if len(dataset) == 0: print(f"[ERROR] Dataset empty: {data_root}"); return
        
    actual_samples = min(num_samples, len(dataset))
    print(f"\n[PHASE 1] True Multiprocessing Benchmark ({actual_samples} samples)...")

    subset = Subset(dataset, range(actual_samples))
    dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_audio_batch)
    
    t_start = time.perf_counter()
    total_batches, total_items = 0, 0
    
    for batch in dataloader:
        total_batches += 1
        items_in_batch = batch.mixture.shape[0]
        total_items += items_in_batch
        print(f"  -> Processed batch {total_batches} ({items_in_batch} samples)")
        
    t_end = time.perf_counter()
    total_time = t_end - t_start
    
    print("\n" + "-"*65)
    print(f"➔ Total Samples: {total_items} | Total Time: {total_time:.2f}s")
    print(f"⚡ ACTUAL THROUGHPUT: {total_items / total_time:.2f} samples / second")
    print("="*65 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Pipeline Testing and Smoke Tests")
    parser.add_argument("--data_root", type=str, required=True, help="Path to preprocessed .pt files")
    parser.add_argument("--idx", type=int, default=0, help="Dataset index to test/visualize")
    parser.add_argument("--benchmark", action="store_true", help="Run full pipeline performance benchmark instead of visualizing")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for the benchmark dataloader")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the benchmark dataloader")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of total samples to process in benchmark")
    args = parser.parse_args()

    if args.benchmark:
        run_pipeline_benchmark(args.data_root, num_samples=args.num_samples, matrix_res=46, batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        visualize_dataloader_batch(args.data_root, idx=args.idx, matrix_res=46)