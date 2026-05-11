import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# ==========================================
# MODULE 1: Acoustic Front-End
# ==========================================
class GeMPoolFreq(nn.Module):
    def __init__(self, p=1.8327, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        p_clamped = self.p.float().clamp(min=1.0, max=10.0)
        
        x_clamped = x_fp32.clamp(min=self.eps)
        out = x_clamped.pow(p_clamped).mean(dim=2).pow(1.0 / p_clamped)
        
        return out.to(x.dtype)

class GeMPooledSampleID(nn.Module):
    def __init__(self, embed_dim=2048,  apply_dora=True, dora_rank=128, **kwargs):
        super().__init__()
        from sampleid.inference import SampleID
        
        self.sampleid = SampleID.load_checkpoint()
        base_encoder = self.sampleid.encoder
        
        if apply_dora and HAS_PEFT and dora_rank > 0:

            target_modules = []
            for name, module in base_encoder.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                    if any(k in name for k in ["backbone", "frontend", "skip", "proj"]):
                        target_modules.append(name)
            

            rank_pattern = {
                "backbone.13": 128, "encoder.backbone.13": 128,
                "backbone.14": 128, "encoder.backbone.14": 128,
                "backbone.15": 128, "encoder.backbone.15": 128,
                "proj": 128, "encoder.proj": 128
            }
            
            config = LoraConfig(
                r=dora_rank, 
                lora_alpha=dora_rank * 2,           
                target_modules=target_modules, 
                rank_pattern=rank_pattern,
                lora_dropout=0.05,
                bias="none", 
                use_dora=True, 
                init_lora_weights="gaussian"
            )
            self.encoder_wrapper = get_peft_model(base_encoder, config)
            self.actual_encoder = self.encoder_wrapper.base_model.model
        else:
            self.encoder_wrapper = base_encoder
            self.actual_encoder = base_encoder
            for param in self.encoder_wrapper.parameters():
                param.requires_grad = False
            
        self.gem_pool = GeMPoolFreq(p=1.8327)

    def forward(self, videos):
        x = videos
        vqt = self.sampleid.transform(x)
        
        crop_size = getattr(self.sampleid, 'crop_size', 4)
        if crop_size > 0:
            vqt = vqt[:, crop_size: -crop_size]
            
        if vqt.ndim == 3: x = vqt.unsqueeze(1)
        elif vqt.ndim == 4: x = vqt
        
        if hasattr(self.actual_encoder, 'prenorm'): x = self.actual_encoder.prenorm(x)
        x = self.actual_encoder.frontend(x)
        for block in self.actual_encoder.backbone: x = block(x)
            
        emb = self.gem_pool(x) 
        
        return {'outputs': emb.permute(0, 2, 1)}

# ==========================================
# MODULE 2: Depthwise Conv1d Projector (Upgraded to TCN)
# ==========================================
class TCNBlock(nn.Module):
    """
    Adds non-linear capacity and expands the temporal receptive field using dilated 
    depthwise convolutions, preventing the 'linear collapse' of standard chained 1x1 convolutions.
    """
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        # Dynamically calculate GroupNorm groups to avoid divisibility crashes (Audit 3.3)
        gn_groups = 32
        while channels % gn_groups != 0 and gn_groups > 1:
            gn_groups //= 2
            
        self.net = nn.Sequential(
            # Depthwise dilated convolution
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels, bias=False),
            # GroupNorm is stable across varying batch sizes
            nn.GroupNorm(gn_groups, channels),

            nn.GELU(),
            # Pointwise convolution to mix cross-channel features
            nn.Conv1d(channels, channels, 1, bias=True)
        )

    def forward(self, x):
        return x + self.net(x) # Residual connection preserves gradient flow

class Conv1dTemporalProjector(nn.Module):
    def __init__(self, embed_dim=2048, kernel_size=3):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-layer Temporal Convolutional Network (TCN) 
        # increasing dilation to capture long-range dependencies without excessive depth.
        self.tcn = nn.Sequential(
            TCNBlock(embed_dim, kernel_size=kernel_size, dilation=1),
            TCNBlock(embed_dim, kernel_size=kernel_size, dilation=2),
            TCNBlock(embed_dim, kernel_size=kernel_size, dilation=4)
        )

    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        x_temporal = self.tcn(x_permuted)
        
        x_proj = x_temporal.permute(0, 2, 1)
        
        # Geometry centering of embeddings to stabilize training and 
        # improve convergence speed, especially important for contrastive losses.
        x_centered = x_proj - x_proj.mean(dim=1, keepdim=True)
        return x_centered

# ==========================================
# MASTER MODEL
# ==========================================
class SupConAudioAligner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg.get("output_dim", 2048)
        self.backbone = GeMPooledSampleID(embed_dim=embed_dim, apply_dora=True, dora_rank=cfg.get("dora_rank", 128))
        
        self.proj_nce = Conv1dTemporalProjector(embed_dim=embed_dim, kernel_size=3)
        self.proj_supcon = Conv1dTemporalProjector(embed_dim=embed_dim, kernel_size=3)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "gem_pool" not in name: param.requires_grad = False
            
    def unfreeze_backbone(self):
        # 1. Unfreeze distinct custom layer injections explicitly.
        for name, param in self.backbone.named_parameters():
            if any(k in name for k in ["gem_pool", "lora", "lora_magnitude_vector"]):
                param.requires_grad = True
            else: 
                param.requires_grad = False
                

        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm,
                                   nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d, nn.GroupNorm, nn.LayerNorm)):
                for param in module.parameters():
                    param.requires_grad = True

    def train(self, mode=True):
        super().train(mode)

        if mode:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.modules.batchnorm._BatchNorm)):
                    m.eval()

    def forward(self, mix, ref):
        f_mix = self.backbone(mix)['outputs']
        f_ref = self.backbone(ref)['outputs']

        m_nce = self.proj_nce(f_mix)
        r_nce = self.proj_nce(f_ref)
        
        m_supcon = self.proj_supcon(f_mix)
        r_supcon = self.proj_supcon(f_ref)
        
        return m_nce, r_nce, m_supcon, r_supcon