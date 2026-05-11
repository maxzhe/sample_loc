import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from data import SampleIDDataModule
from lightning_module import SampleDetectorLit

@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig):

    print("=" * 60)
    print(" INITIALIZING SINCERE/INFONCE CONTRASTIVE CYCLIC PIPELINE")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    
    # Reproducibility
    pl.seed_everything(cfg.data.seed, workers=True)

    # Initialize Modules (Updated SampleID)
    datamodule = SampleIDDataModule(cfg)
    system = SampleDetectorLit(cfg)

    # Ensure checkpoint directory exists
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    
    # Setup Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="SINCERE-Audio-{epoch:02d}-{val/loss:.3f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    csv_logger = CSVLogger(save_dir=cfg.training.checkpoint_dir, name="csv_logs")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="auto",
        devices="auto",
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=50, 
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )

    # Resume from last checkpoint if available
    last_checkpoint_path = os.path.join(cfg.training.checkpoint_dir, "last.ckpt")
    if os.path.exists(last_checkpoint_path):
        print(f"🔄 Resuming training from checkpoint: {last_checkpoint_path}")
        trainer.fit(system, datamodule=datamodule, ckpt_path=last_checkpoint_path)
    else:
        print("🌱 Starting fresh SINCERE alignment training run.")
        trainer.fit(system, datamodule=datamodule)

if __name__ == "__main__":
    main()