from dataclasses import dataclass
import torch

@dataclass
class AudioSample:
    """
    Strict contract defining exactly what a single CPU Worker thread 
    must output from the Dataset's __getitem__ method.
    """
    mixture: torch.Tensor          # Shape: [T_mix]
    reference: torch.Tensor        # Shape: [T_ref]
    ref_target: torch.Tensor       # Shape: [T_ref]
    ref_bg: torch.Tensor           # Shape: [T_ref]
    bg_ducked: torch.Tensor        # Shape: [T_mix]
    target_only: torch.Tensor      # Shape: [T_mix]
    gt_presence: torch.Tensor      # Shape: [Matrix_Res]
    gt_coords: torch.Tensor        # Shape: [Matrix_Res]
    is_loop: torch.Tensor          # Shape: [1] (Boolean)
    chunk_frames: torch.Tensor     # Shape: [1] (Float)

@dataclass
class AudioBatch:
    """
    Strict contract defining exactly what the PyTorch Lightning Model 
    expects to receive from the DataLoader's collate_fn.
    """
    mixture: torch.Tensor          # Shape: [B, T_mix]
    reference: torch.Tensor        # Shape: [B, T_ref]
    ref_target: torch.Tensor       # Shape: [B, T_ref]
    ref_bg: torch.Tensor           # Shape: [B, T_ref]
    bg_ducked: torch.Tensor        # Shape: [B, T_mix]
    target_only: torch.Tensor      # Shape: [B, T_mix]
    gt_presence: torch.Tensor      # Shape: [B, Matrix_Res]
    gt_coords: torch.Tensor        # Shape: [B, Matrix_Res]
    is_loop: torch.Tensor          # Shape: [B]
    chunk_frames: torch.Tensor     # Shape: [B]