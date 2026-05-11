"""
=========================================================================================
DUAL-CONTRASTIVE LOSS FRAMEWORK FOR AUDIO ALIGNMENT
=========================================================================================
This module implements a dual-loss architecture to solve the core "Engineering Paradox" 
of temporal audio alignment, specifically when dealing with repeating/looping sounds:

1. Goal A (Localization): Finding the exact micro-second a sound occurs ("Where is it?").
   - Handled by `DenseCrossBatchInfoNCE`. It strictly penalizes timing errors to create 
     a needle-like temporal peak.
2. Goal B (Representation): Recognizing that repeating loops sound identical ("What is it?").
   - Handled by `SINCERELoss`. It prevents identical sounds from mathematically repelling 
     each other, ensuring the network learns true acoustic semantics.

By running these losses in tandem (total_loss = l_supcon + l_nce), the model achieves 
sub-frame temporal precision without succumbing to 'Destructive Intra-Class Repulsion' 
(where traditional contrastive learning destroys semantic understanding by pushing 
identical repeating beats apart).
=========================================================================================
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import SupConAudioAligner
from datatypes import AudioBatch

class DenseCrossBatchInfoNCE(nn.Module):
    """
    Dense Cross-Batch InfoNCE Loss (The "Where is it?" Localization Loss).
    
    Goal: Perfect temporal sequence alignment. It forms an ultra-sharp probability peak
    over the exact, true chronological timestamp, ignoring acoustic similarities elsewhere.
    
    Research & Architecture Points:
    - Anchor (`m_centered`): A noisy mixture frame that is strictly mathematically aligned 
      to the underlying grid.
    - The Positive (1-to-1): Exactly one true fractional coordinate on the reference timeline.
      Instead of strict binary targets, it uses a Gaussian bell curve to provide smooth,
      differentiable gradients for being fractions of a frame away.
    - The Negatives (1-to-N): Every single other reference frame generated across the
      entire batch (`B * T_ref`).
    - Focal Repulsion (Sidelobe Suppression): Actively adds a mathematical penalty (`focal_penalty`)
      to immediate temporal neighbors (frames 1 to 3 steps away) to sharpen the peak and
      prevent blurry temporal predictions.
    - Loop-Aware Math: Uses modulo arithmetic (`linear_dist % c_frames`) to wrap around
      looping audio, applying the focal repulsion penalty to EVERY repetition of the loop.
    """
    def __init__(self, temperature=0.07, alignment_tolerance=0.05, relaxed_alignment_tolerance=0.5, min_valid_anchors=4):
        super().__init__()
        # Learnable temperature parameter for contrastive scaling
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))
        
        # Tolerances dictate how far a frame can drift from the exact integer index before we consider it "unaligned"
        self.alignment_tolerance = alignment_tolerance 
        self.relaxed_alignment_tolerance = relaxed_alignment_tolerance
        self.min_valid_anchors = min_valid_anchors

    def compute_gaussian_soft_labels(self, sim_matrix, exact_r_idx_float, T_ref, b_idx, sigma=0.1):
        """
        Creates a target distribution using Gaussian soft-labels around the exact floating-point 
        ground truth index. This solves the "partial credit" problem for continuous temporal tracking.
        """
        N_anchors = exact_r_idx_float.size(0)
        device = sim_matrix.device
        
        # 1. Create a temporal grid: [1, T_ref]
        grid = torch.arange(T_ref, device=device, dtype=torch.float32).unsqueeze(0)
        
        # 2. Calculate squared distance from exact floating-point coordinate
        dist_sq = (grid - exact_r_idx_float.unsqueeze(1)) ** 2
        
        # 3. Apply Gaussian decay and normalize so probabilities sum to 1
        soft_labels = torch.exp(-dist_sq / (2 * (sigma ** 2)))
        soft_labels = soft_labels / soft_labels.sum(dim=1, keepdim=True)
        
        # 4. Map these localized soft labels into the massive global similarity matrix
        full_soft_labels = torch.zeros_like(sim_matrix)
        for i in range(N_anchors):
            batch_offset = b_idx[i] * T_ref
            full_soft_labels[i, batch_offset : batch_offset + T_ref] = soft_labels[i]
            
        return full_soft_labels

    def forward(self, m_centered, r_centered, gt_presence, gt_coords, w_mix=None, w_ref=None, max_sec=15.0, is_loop=None, chunk_frames=None):
        # Embeddings format: [Batch, Time, Dim]
        B, T_mix, D = m_centered.shape
        _, T_ref, _ = r_centered.shape
        device = m_centered.device
        
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=0.5)

        # Convert ground truth coordinates from seconds to fractional frame indices
        sec_per_frame = max_sec / T_ref
        exact_ref_indices_float = gt_coords / sec_per_frame
        ref_indices_int = torch.round(exact_ref_indices_float)
        
        raw_diff = torch.abs(exact_ref_indices_float - ref_indices_int)
        align_err = torch.minimum(raw_diff, T_ref - raw_diff)

        # -----------------------------------------------------------------
        # Anchor Selection (Hard vs. Relaxed Filtering)
        # InfoNCE requires strict grid alignment to calculate distance accurately
        # -----------------------------------------------------------------
        is_perfectly_aligned = align_err <= self.alignment_tolerance
        active_mask = (gt_presence > 0.5) & is_perfectly_aligned
        b_idx, t_idx = torch.where(active_mask)
        
        if len(b_idx) < self.min_valid_anchors:
            is_perfectly_aligned = align_err <= self.relaxed_alignment_tolerance
            active_mask = (gt_presence > 0.5) & is_perfectly_aligned
            b_idx, t_idx = torch.where(active_mask)

        if len(b_idx) == 0:
            dummy_loss = (m_centered.sum() * 0.0) + (r_centered.sum() * 0.0)
            return dummy_loss, 0, {
                "pos_sim": 0.0, "neg_sim": 0.0, "pslr_db": 0.0, "near_neighbor_sim": 0.0,
                "pos_per_anchor": 0.0, "neg_per_anchor": 0.0, "soft_neg_per_anchor": 0.0,
                "avg_confidence_gate": 0.0, "global_retrieval_acc": 0.0, "global_mrr": 0.0, "global_top3_acc": 0.0
            }

        r_idx = torch.clamp(ref_indices_int[b_idx, t_idx].long(), 0, T_ref - 1)
        exact_anchors_float = exact_ref_indices_float[b_idx, t_idx]

        anchor_weights = (w_mix[b_idx, t_idx] * w_ref[b_idx, r_idx]) if w_mix is not None and w_ref is not None else torch.ones_like(b_idx, dtype=torch.float32)
        
        anchors = F.normalize(m_centered[b_idx, t_idx, :], p=2, dim=-1, eps=1e-8) 
        all_refs = F.normalize(r_centered.contiguous().view(B * T_ref, D), p=2, dim=-1, eps=1e-8)

        # The 1-to-N Massive Global similarity matrix: [Num_Anchors, B * T_ref]
        sim_matrix = torch.matmul(anchors, all_refs.T) / tau 

        # -----------------------------------------------------------------
        # Temporal Focal Repulsion (Sidelobe Suppression)
        # -----------------------------------------------------------------
        col_batch_idx = torch.arange(B * T_ref, device=device) // T_ref
        col_time_idx = torch.arange(B * T_ref, device=device) % T_ref
        same_pair_mask = (b_idx.unsqueeze(1) == col_batch_idx.unsqueeze(0)) 
        linear_dist = torch.abs(r_idx.unsqueeze(1) - col_time_idx.unsqueeze(0)).float()
        
        # Loop-Aware Math: Wrapping distances for periodic audio using modulo
        if is_loop is not None and chunk_frames is not None:
            loop_mask = is_loop[b_idx]
            c_frames = torch.clamp(chunk_frames[b_idx].unsqueeze(1), min=1.0)
            mod_dist = linear_dist % c_frames
            cyclic_dist = torch.minimum(mod_dist, c_frames - mod_dist)
            temporal_dist = torch.where(loop_mask.unsqueeze(1), cyclic_dist, linear_dist)
        else:
            temporal_dist = linear_dist

        # Apply focal penalty to immediate neighbors to force a sharp temporal peak
        near_neighbor_mask = (temporal_dist > 0) & (temporal_dist <= 3.0) & same_pair_mask
        focal_repulsion_margin = 0.25 
        focal_penalty = (focal_repulsion_margin * near_neighbor_mask.float()) / tau
        
        sim_matrix = sim_matrix + focal_penalty
        
        # -----------------------------------------------------------------
        # Diagnostic Metrics Collection
        # -----------------------------------------------------------------
        flat_pos_idx = b_idx * T_ref + r_idx
        row_idx = torch.arange(len(b_idx), device=device)

        with torch.no_grad():
            raw_sims = (sim_matrix - focal_penalty) * tau
            pos_sim = raw_sims[row_idx, flat_pos_idx].mean().item()
            
            if len(row_idx) > 0:
                sims_by_track = raw_sims.view(len(row_idx), B, T_ref)
                track_max_sims = sims_by_track.max(dim=2)[0] 
                gt_track_sims = track_max_sims[row_idx, b_idx].unsqueeze(1)
                ranks = (track_max_sims > gt_track_sims).sum(dim=1) + 1
                
                global_retrieval_acc = (ranks == 1).float().mean().item()  
                global_mrr = (1.0 / ranks.float()).mean().item()           
                global_top3_acc = (ranks <= 3).float().mean().item()       
            else:
                global_retrieval_acc = global_mrr = global_top3_acc = 0.0
            
            neg_mask = (raw_sims > -1e8)
            neg_mask[row_idx, flat_pos_idx] = False
            valid_neg_sims = raw_sims[neg_mask]
            neg_sim = valid_neg_sims.mean().item() if valid_neg_sims.numel() > 0 else 0.0
            
            has_neighbors = near_neighbor_mask.sum(dim=1) > 0
            if has_neighbors.any():
                masked_raw_sim_abs = raw_sims.abs().masked_fill(~near_neighbor_mask, 0.0)
                max_sidelobe_abs = masked_raw_sim_abs[has_neighbors].max(dim=1)[0]
                peak_sim_val = raw_sims[row_idx, flat_pos_idx][has_neighbors]
                
                pslr_db = 10.0 * torch.log10((peak_sim_val**2 + 1e-9) / (max_sidelobe_abs**2 + 1e-9)).mean().item()
                near_neighbor_sim = raw_sims[near_neighbor_mask].mean().item()
                soft_neg_per_anchor = near_neighbor_mask.sum(dim=1).float().mean().item()
            else:
                pslr_db = near_neighbor_sim = soft_neg_per_anchor = 0.0
                
            pos_per_anchor, neg_per_anchor = 1.0, float((B * T_ref) - 1)
            avg_conf = anchor_weights.mean().item()

        # Final KL Divergence optimization against Gaussian Soft Labels
        log_probs = F.log_softmax(sim_matrix, dim=1)
        soft_targets = self.compute_gaussian_soft_labels(sim_matrix, exact_anchors_float, T_ref, b_idx, sigma=0.1)
        
        raw_kl_div = F.kl_div(log_probs, soft_targets, reduction='none')
        loss_per_anchor = raw_kl_div.sum(dim=1)
        
        weighted_loss = loss_per_anchor * anchor_weights.detach()
        loss = weighted_loss.mean()
        
        return loss, len(b_idx), {
            "pos_sim": pos_sim, "neg_sim": neg_sim, "pslr_db": pslr_db, "near_neighbor_sim": near_neighbor_sim,
            "pos_per_anchor": pos_per_anchor, "neg_per_anchor": neg_per_anchor, "soft_neg_per_anchor": soft_neg_per_anchor,
            "avg_confidence_gate": avg_conf, "global_retrieval_acc": global_retrieval_acc,
            "global_mrr": global_mrr, "global_top3_acc": global_top3_acc
        }


class SINCERELoss(nn.Module):
    """
    Supervised InfoNCE REvisited [SINCERE] (The "What is it?" Representation Loss).
    
    Goal: Acoustic semantic representation. It ensures the model understands what the audio
    actually is, clustering identical sounds regardless of when they happen.
    
    Research & Architecture Points:
    - Problem Solved (Destructive Intra-Class Repulsion): Standard contrastive loss pushes
      identical looping sounds apart because they occur at different times. SINCERE eliminates
      this mathematical penalty.
    - Anchor (`m_centered`): A noisy mixture frame containing the target audio.
    - The Positives (Many-to-Many): Uses a deterministic mathematical ID (`anchor_classes`).
      Any frame sharing this ID (like identical repeating loops) is treated as a positive.
    - The Masked Denominator (The SINCERE Core): Uses `full_pos_mask` to physically extract
      all identical sounds from the contrastive negative denominator pool. The model incurs
      zero penalty for placing identical snare hits from different timestamps close together.
    - Hard Background Mining: Explicitly concatenates shifted background noise (`sim_ab`)
      into the negative matrix. This forces the model to actively push background interference
      away from the target semantic feature.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)))

    def forward(self, m_centered, r_centered, gt_presence, gt_coords, max_sec, bg_centered=None, w_mix=None, w_ref=None):
        B, T_mix, D = m_centered.shape
        _, T_ref, _ = r_centered.shape
        device = m_centered.device
        
        tau = torch.exp(self.log_tau).clamp(min=0.01, max=0.5)

        # Calculate integer frame indices
        sec_per_frame = max_sec / T_ref
        ref_indices_int = torch.round(gt_coords / sec_per_frame)
        active_mask = (gt_presence > 0.5)
        b_idx, t_idx = torch.where(active_mask)
        
        if len(b_idx) == 0: 
            dummy_loss = (m_centered.sum() * 0.0) + (r_centered.sum() * 0.0)
            return dummy_loss, {
                "pos_sim": 0.0, "neg_sim": 0.0, "hard_bg_sim": 0.0,
                "pos_per_anchor": 0.0, "neg_per_anchor": 0.0, "hard_bg_per_anchor": 0.0
            }
        
        r_idx = torch.clamp(ref_indices_int[b_idx, t_idx].long(), 0, T_ref - 1)
        anchor_weights = (w_mix[b_idx, t_idx] * w_ref[b_idx, r_idx]) if w_mix is not None and w_ref is not None else torch.ones_like(b_idx, dtype=torch.float32)

        # -----------------------------------------------------------------
        # Representation Masking & Matrix Setup
        # -----------------------------------------------------------------
        anchors = F.normalize(m_centered[b_idx, t_idx, :], p=2, dim=-1, eps=1e-8)
        
        # The 'Class' acts as the deterministic mathematical ID for identical audio chunks
        anchor_classes = b_idx * T_ref + r_idx

        # 1. Self-similarity logic (Anchor-to-Anchor)
        sim_aa = torch.matmul(anchors, anchors.T) / tau
        N = len(anchors)
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_aa.masked_fill_(self_mask, -1e9) # Prevent an anchor from matching itself

        class_mask = (anchor_classes.unsqueeze(1) == anchor_classes.unsqueeze(0))
        pos_mask_aa = class_mask & ~self_mask

        # 2. Reference similarity logic (Anchor-to-Reference)
        unique_classes, inverse_indices = torch.unique(anchor_classes, return_inverse=True)
        unique_b = unique_classes // T_ref
        unique_r = unique_classes % T_ref
        
        ref_anchors = F.normalize(r_centered[unique_b, unique_r, :], p=2, dim=-1, eps=1e-8)
        sim_ar = torch.matmul(anchors, ref_anchors.T) / tau
        pos_mask_ar = (inverse_indices.unsqueeze(1) == torch.arange(len(unique_classes), device=device).unsqueeze(0))

        # Combine Anchor-to-Anchor and Anchor-to-Reference blocks into the "Frankenstein" Matrix
        sim_matrix = torch.cat([sim_aa, sim_ar], dim=1)
        full_pos_mask = torch.cat([pos_mask_aa, pos_mask_ar], dim=1)

        # -----------------------------------------------------------------
        # Hard Background Mining (sim_ab)
        # -----------------------------------------------------------------
        bg_sim_val = 0.0
        num_bg_anchors = 0.0
        if bg_centered is not None:
            # Randomly shift background in time so the network doesn't memorize static correlations
            B_bg, T_bg, D_bg = bg_centered.shape
            shift_amount = torch.randint(1, T_bg // 2, (1,)).item()
            bg_centered_shifted = torch.roll(bg_centered, shifts=-shift_amount, dims=1)
            
            bg_anchors = F.normalize(bg_centered_shifted[b_idx, t_idx, :], p=2, dim=-1, eps=1e-8)
            num_bg_anchors = float(bg_anchors.shape[0])
            
            raw_bg_dot = torch.matmul(anchors, bg_anchors.T)
            bg_sim_val = raw_bg_dot.mean().item()
            
            sim_ab = raw_bg_dot / tau
            overlap_attenuation = anchor_weights.unsqueeze(1).detach()
            sim_ab = sim_ab * overlap_attenuation

            # Background frames are explicitly labeled as NEGATIVE targets to force orthogonalization
            pos_mask_ab = torch.zeros_like(sim_ab, dtype=torch.bool, device=device)
            sim_matrix = torch.cat([sim_matrix, sim_ab], dim=1)
            full_pos_mask = torch.cat([full_pos_mask, pos_mask_ab], dim=1)
            
        # -----------------------------------------------------------------
        # Masked Softmax Denominator Calculation (The "SINCERE" Core)
        # -----------------------------------------------------------------
        with torch.no_grad():
            raw_supcon_sims = sim_matrix * tau
            valid_pos_sims = raw_supcon_sims[full_pos_mask]
            supcon_pos_sim = valid_pos_sims.mean().item() if valid_pos_sims.numel() > 0 else 0.0
            
            valid_neg_mask = (~full_pos_mask) & (raw_supcon_sims > -1e8)
            valid_neg_sims = raw_supcon_sims[valid_neg_mask]
            supcon_neg_sim = valid_neg_sims.mean().item() if valid_neg_sims.numel() > 0 else 0.0

        max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0]
        exp_sim = torch.exp(sim_matrix - max_sim)
        
        # KEY: We sum ONLY the negative pairs. Positive pairs are extracted via `~full_pos_mask`.
        # This completely eliminates Destructive Intra-Class Repulsion.
        sum_neg = (exp_sim * (~full_pos_mask).float()).sum(dim=1, keepdim=True)
        log_prob_pos = (sim_matrix - max_sim) - torch.log(exp_sim + sum_neg + 1e-9)
        
        num_pos = full_pos_mask.sum(dim=1)
        valid_rows = num_pos > 0
        
        with torch.no_grad():
            if valid_rows.any():
                supcon_pos_per_anchor = num_pos[valid_rows].float().mean().item()
                supcon_neg_per_anchor = (~full_pos_mask)[valid_rows].sum(dim=1).float().mean().item()
            else:
                supcon_pos_per_anchor = supcon_neg_per_anchor = 0.0

        metrics = {
            "pos_sim": supcon_pos_sim, "neg_sim": supcon_neg_sim, "hard_bg_sim": bg_sim_val,
            "pos_per_anchor": supcon_pos_per_anchor, "neg_per_anchor": supcon_neg_per_anchor, "hard_bg_per_anchor": num_bg_anchors
        }
        
        if not valid_rows.any(): 
            return torch.tensor(0.0, device=device, requires_grad=True), metrics
            
        loss_per_anchor = - (full_pos_mask[valid_rows].float() * log_prob_pos[valid_rows]).sum(dim=1) / num_pos[valid_rows].float()
        weighted_loss = loss_per_anchor * anchor_weights[valid_rows].detach()
        
        return weighted_loss.mean(), metrics


class SampleDetectorLit(pl.LightningModule):
    """
    PyTorch Lightning Module encapsulating the model, losses, and curriculum learning.
    Manages the interplay between InfoNCE, SINCERE, and dynamic batch constraints.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lr = cfg.training.lr
        self.warmup_epochs = self.cfg.training.get('warmup_epochs', 0)
        self.health_log_freq = self.cfg.training.get('health_log_freq', 50) 
        
        model_cfg = {
            "output_dim": cfg.model.get("dim", 2048),
            "dora_rank": cfg.model.get("dora_rank", 128)
        }
        self.model = SupConAudioAligner(cfg=model_cfg)
        
        safeguards = self.cfg.training.get('safeguards', {})
        self.contrastive_loss = DenseCrossBatchInfoNCE(
            alignment_tolerance=safeguards.get('alignment_tolerance', 0.05),
            relaxed_alignment_tolerance=safeguards.get('relaxed_alignment_tolerance', 0.5),
            min_valid_anchors=safeguards.get('min_valid_anchors', 4)
        )
        self.supcon_loss = SINCERELoss()
        self.current_max_sec = self.cfg.data.get('source_sec', 15.0)

    def _apply_data_curriculum(self):
        epoch = self.current_epoch
        cur = self.cfg.training.get('curriculum', {})
        p1 = cur.get('phase1_epochs', 15)
        p2 = cur.get('phase2_epochs', 35)
        p3 = cur.get('phase3_epochs', 50)

        total_epochs = p1 + p2 + p3
        global_prog = min(1.0, epoch / max(1, total_epochs))

        length = self.cfg.data.get('source_sec', 15.0)
        base_loop_prob = self.cfg.data.get('looping_prob', 0.5)
        loop = base_loop_prob * global_prog

        snr_min, snr_max = self.cfg.data.get('snr_db_range', (-10.0, 0.0))
        target_base_snr = (snr_min + snr_max) / 2.0
        target_variance = (snr_max - snr_min) / 2.0

        start_base_snr, start_variance = 1.0, 1.0
        snr_dist = None
        
        if epoch < p1 + p2:
            prog_snr = epoch / max(1, p1 + p2)
            current_base = start_base_snr - (start_base_snr - target_base_snr) * prog_snr
            current_variance = start_variance + (target_variance - start_variance) * prog_snr
            snr = (current_base - current_variance, current_base + current_variance)
            
            if epoch < p1:
                dist = 0.0
            else:
                prog_dist = (epoch - p1) / max(1, p2)
                dist = 0.4 + 0.4 * prog_dist 
        else:
            snr = (snr_min, snr_max) 
            dist = 0.8
            snr_dist = cur.get('phase3_snr_distribution', None)

        self.trainer.datamodule.update_curriculum(snr, length, loop, dist, snr_dist)
        self.current_max_sec = length

    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_epochs:
            self.model.freeze_backbone()
            self.log("debug/backbone_frozen", 1.0, sync_dist=True)
        else:
            self.model.unfreeze_backbone()
            self.log("debug/backbone_frozen", 0.0, sync_dist=True)
            
        self._apply_data_curriculum()

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for module in self.model.modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, 
                                       torch.nn.modules.batchnorm._BatchNorm)):
                    module.eval()

    def _align_1d_targets(self, tensor_1d, target_len, is_coords=False):
        B = tensor_1d.shape[0]
        if tensor_1d.shape[-1] != target_len:
            mode = 'linear' if is_coords else 'nearest'
            align_corners = False if is_coords else None
            if is_coords:
                return F.interpolate(tensor_1d.view(B, 1, -1), size=target_len, mode=mode, align_corners=align_corners).squeeze(1)
            else:
                return F.interpolate(tensor_1d.view(B, 1, -1), size=target_len, mode=mode).squeeze(1)
        return tensor_1d

    def _plot_validation_grid(self, gt_presences, sim_matrix, gt_coords_1d, epoch, num_samples=20):
        B = gt_presences.shape[0]
        num_to_plot = min(num_samples, B)
        fig, axes = plt.subplots(num_to_plot, 1, figsize=(6, 3 * num_to_plot))
        if num_to_plot == 1: axes = np.array([axes])

        for i in range(num_to_plot):
            ax_sim = axes[i]
            gt = gt_presences[i].detach().cpu().numpy()
            raw_sim = sim_matrix[i].detach().cpu().numpy()
            gt_coords = gt_coords_1d[i].detach().cpu().numpy()
            
            T_len = raw_sim.shape[0]
            sec_per_frame = self.current_max_sec / T_len
            gt_coord_idx = gt_coords / sec_per_frame
            
            ax_sim.imshow(raw_sim.T, origin='lower', aspect='auto', cmap='viridis', alpha=0.9)
            ax_sim.plot(np.where(gt > 0.5, gt_coord_idx, np.nan), color='cyan', linewidth=3, label="GT Line")
            ax_sim.set_title(f"Sample {i+1} - Raw Cosine Similarity (DMC)")

        plt.tight_layout()
        save_dir = os.path.join(self.cfg.training.checkpoint_dir, "val_plots")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"val_epoch_{epoch:03d}_grid.png"), bbox_inches='tight')
        plt.close(fig)

    def _calculate_gpu_confidence_gate(self, target_only, bg_ducked, T_tokens):
        """
        Calculates observability masking by measuring physical SNR inside the tensors,
        protecting the model from penalizing anchors hidden below the noise floor.
        """
        tgt_sq = target_only.unsqueeze(1) ** 2
        bg_sq = bg_ducked.unsqueeze(1) ** 2
        
        tgt_energy = F.adaptive_avg_pool1d(tgt_sq, T_tokens).squeeze(1)
        bg_energy = F.adaptive_avg_pool1d(bg_sq, T_tokens).squeeze(1)
        
        snr_db = 10 * torch.log10((tgt_energy + 1e-9) / (bg_energy + 1e-9))
        
        theta = self.cfg.training.get('observability', {}).get('theta_db', -15.0)
        beta = self.cfg.training.get('observability', {}).get('beta_temp', 3.0)
        
        confidence_weights = torch.sigmoid((snr_db - theta) / beta)
        return confidence_weights

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % self.health_log_freq != 0: return
        grad_sq_sums = {"total": 0.0, "backbone": 0.0, "projector": 0.0}
        param_sq_sums = {"total": 0.0, "backbone": 0.0, "projector": 0.0}
        
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            param_sq = param.data.norm(2).item() ** 2
            param_sq_sums["total"] += param_sq
            if "backbone" in name: param_sq_sums["backbone"] += param_sq
            elif "projector" in name: param_sq_sums["projector"] += param_sq
                
            if param.grad is not None:
                grad_sq = param.grad.data.norm(2).item() ** 2
                grad_sq_sums["total"] += grad_sq
                if "backbone" in name: grad_sq_sums["backbone"] += grad_sq
                elif "projector" in name: grad_sq_sums["projector"] += grad_sq

        log_dict = {f"grad_norm/{k}": v ** 0.5 for k, v in grad_sq_sums.items() if v > 0}
        log_dict.update({f"param_norm/{k}": v ** 0.5 for k, v in param_sq_sums.items() if v > 0})
        if log_dict: self.log_dict(log_dict, prog_bar=False)

    def training_step(self, batch: AudioBatch, batch_idx: int):
        mix = batch.mixture
        ref = batch.reference
        bg_ducked = batch.bg_ducked
        target_only = batch.target_only
        ref_target = batch.ref_target
        ref_bg = batch.ref_bg
        
        m_nce, r_nce, m_supcon, r_supcon = self.model(mix, ref)

        with torch.no_grad():
            _, _, bg_supcon, _ = self.model(bg_ducked, ref)

        active_lambda_supcon = self.cfg.training.loss_weights.get('lambda_supcon', 1.0)

        seq_len = m_nce.shape[1]
        gt_presence = self._align_1d_targets(batch.gt_presence, target_len=seq_len, is_coords=False)
        gt_coords = self._align_1d_targets(batch.gt_coords, target_len=seq_len, is_coords=True)
        
        w_mix = self._calculate_gpu_confidence_gate(target_only, bg_ducked, seq_len)
        w_ref = self._calculate_gpu_confidence_gate(ref_target, ref_bg, seq_len)
        
        l_nce, active_frames, nce_metrics = self.contrastive_loss(
            m_nce, r_nce, gt_presence, gt_coords, w_mix=w_mix, w_ref=w_ref,
            max_sec=self.current_max_sec, is_loop=batch.is_loop, chunk_frames=batch.chunk_frames
        )

        l_supcon, supcon_metrics = self.supcon_loss(
            m_supcon, r_supcon, gt_presence, gt_coords, max_sec=self.current_max_sec, 
            bg_centered=bg_supcon, w_mix=w_mix, w_ref=w_ref
        )

        l_supcon_scaled = active_lambda_supcon * l_supcon
        l_nce_scaled = self.cfg.training.loss_weights.get('lambda_infonce', 1.0) * l_nce
        
        # SYNERGISTIC OPTIMIZATION
        # InfoNCE enforces tight temporal accuracy, SINCERE builds unified semantic clusters
        total_loss = l_supcon_scaled + l_nce_scaled

        self.log_dict({
            "train/loss": total_loss, "train/supcon": l_supcon, "train/infonce": l_nce,
            "debug/learned_tau_supcon": torch.exp(self.supcon_loss.log_tau).item(), 
            "debug/active_contrastive_anchors": float(active_frames),
            "insight/avg_confidence_gate": float(nce_metrics["avg_confidence_gate"]),
            "insight/infonce_pos_sim": float(nce_metrics["pos_sim"]),
            "insight/infonce_neg_sim": float(nce_metrics["neg_sim"]),
            "insight/infonce_near_neighbor_sim": float(nce_metrics["near_neighbor_sim"]),
            "insight/infonce_pslr_db": float(nce_metrics["pslr_db"]),
            "insight/infonce_global_acc": float(nce_metrics["global_retrieval_acc"]),
            "insight/infonce_global_mrr": float(nce_metrics["global_mrr"]),        
            "insight/infonce_global_top3": float(nce_metrics["global_top3_acc"]),  
            "insight/supcon_hard_bg_sim": float(supcon_metrics["hard_bg_sim"]),
            "insight/supcon_pos_sim": float(supcon_metrics["pos_sim"]),
            "insight/supcon_neg_sim": float(supcon_metrics["neg_sim"])
        }, prog_bar=True, on_step=True)

        return total_loss

    def validation_step(self, batch: AudioBatch, batch_idx: int):
        mix = batch.mixture
        ref = batch.reference
        bg_ducked = batch.bg_ducked
        target_only = batch.target_only
        ref_target = batch.ref_target
        ref_bg = batch.ref_bg
        
        m_nce, r_nce, m_supcon, r_supcon = self.model(mix, ref)

        with torch.no_grad(): 
             _, _, bg_supcon, _ = self.model(bg_ducked, ref)

        seq_len = m_nce.shape[1]
        gt_presence = self._align_1d_targets(batch.gt_presence, target_len=seq_len, is_coords=False)
        gt_coords = self._align_1d_targets(batch.gt_coords, target_len=seq_len, is_coords=True)
        
        w_mix = self._calculate_gpu_confidence_gate(target_only, bg_ducked, seq_len)
        w_ref = self._calculate_gpu_confidence_gate(ref_target, ref_bg, seq_len)

        l_nce, _, nce_metrics = self.contrastive_loss(
            m_nce, r_nce, gt_presence, gt_coords, w_mix=w_mix, w_ref=w_ref,
            max_sec=self.current_max_sec, is_loop=batch.is_loop, chunk_frames=batch.chunk_frames
        )
        l_supcon, supcon_metrics = self.supcon_loss(
            m_supcon, r_supcon, gt_presence, gt_coords, max_sec=self.current_max_sec, 
            bg_centered=bg_supcon, w_mix=w_mix, w_ref=w_ref
        )
        
        m_norm = F.normalize(m_nce, p=2, dim=-1, eps=1e-8)
        r_norm = F.normalize(r_nce, p=2, dim=-1, eps=1e-8)
        sim_matrix = torch.bmm(m_norm, r_norm.transpose(1, 2))
        
        if batch_idx == 0:
            self._plot_validation_grid(gt_presence, sim_matrix, gt_coords, self.current_epoch)

        self.log_dict({
            "val/loss": l_nce + l_supcon, 
            "val/supcon": l_supcon, "val/infonce": l_nce,
            "val_insight/avg_confidence_gate": float(nce_metrics["avg_confidence_gate"]),
            "val_insight/infonce_pslr_db": float(nce_metrics["pslr_db"]),
            "val_insight/infonce_global_acc": float(nce_metrics["global_retrieval_acc"]),
            "val_insight/infonce_global_mrr": float(nce_metrics["global_mrr"]),        
            "val_insight/infonce_global_top3": float(nce_metrics["global_top3_acc"]),  
            "val_insight/supcon_hard_bg_sim": float(supcon_metrics["hard_bg_sim"]),
            "val_insight/supcon_pos_sim": float(supcon_metrics["pos_sim"]),
            "val_insight/supcon_neg_sim": float(supcon_metrics["neg_sim"])
        }, sync_dist=True)

        return l_nce + l_supcon

    def configure_optimizers(self):
        base_decay, base_no_decay, dora_decay, dora_no_decay = [], [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad: continue
            
            is_no_decay = any(k in name for k in ["bias", "norm", "bn", "inst", "lora_magnitude_vector", "log_tau"])
            is_dora = "lora" in name
            
            if is_dora: (dora_no_decay if is_no_decay else dora_decay).append(param)
            else: (base_no_decay if is_no_decay else base_decay).append(param)
                
        dora_lr = self.lr * 0.2
        
        optimizer = optim.AdamW([
            {"params": base_decay, "weight_decay": 1e-4, "lr": self.lr},
            {"params": base_no_decay, "weight_decay": 0.0, "lr": self.lr},
            {"params": dora_decay, "weight_decay": 1e-4, "lr": dora_lr},
            {"params": dora_no_decay, "weight_decay": 0.0, "lr": dora_lr}
        ])
        
        return [optimizer], [optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs)]