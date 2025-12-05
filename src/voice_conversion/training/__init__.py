"""Training utilities for voice conversion models."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from ..data import VoiceConversionDataModule
from ..models import CycleGANVC, VAEVC, HiFiGANVocoder
from ..utils import get_device, get_model_size

logger = logging.getLogger(__name__)


class VoiceConversionTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for voice conversion models."""
    
    def __init__(
        self,
        model: Union[CycleGANVC, VAEVC],
        config: DictConfig,
        vocoder: Optional[HiFiGANVocoder] = None,
    ):
        """Initialize trainer.
        
        Args:
            model: Voice conversion model.
            config: Training configuration.
            vocoder: Optional vocoder for audio generation.
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.model = model
        self.config = config
        self.vocoder = vocoder
        
        # Log model size
        model_size = get_model_size(model)
        logger.info(f"Model size: {model_size['total_parameters']:,} parameters")
        logger.info(f"Model size: {model_size['total_size_mb']:.2f} MB")
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and schedulers.
        
        Returns:
            Dictionary containing optimizers and schedulers.
        """
        # Generator optimizer
        g_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training.lr,
            betas=(self.config.training.beta1, self.config.training.beta2),
        )
        
        # Discriminator optimizer (for adversarial models)
        if hasattr(self.model, 'D_source'):
            d_optimizer = optim.Adam(
                list(self.model.D_source.parameters()) + list(self.model.D_target.parameters()),
                lr=self.config.training.lr,
                betas=(self.config.training.beta1, self.config.training.beta2),
            )
            
            # Schedulers
            g_scheduler = optim.lr_scheduler.StepLR(
                g_optimizer,
                step_size=self.config.training.lr_decay_step,
                gamma=self.config.training.lr_decay_gamma,
            )
            d_scheduler = optim.lr_scheduler.StepLR(
                d_optimizer,
                step_size=self.config.training.lr_decay_step,
                gamma=self.config.training.lr_decay_gamma,
            )
            
            return {
                "optimizer": [g_optimizer, d_optimizer],
                "lr_scheduler": [g_scheduler, d_scheduler],
            }
        else:
            # VAE model
            scheduler = optim.lr_scheduler.StepLR(
                g_optimizer,
                step_size=self.config.training.lr_decay_step,
                gamma=self.config.training.lr_decay_gamma,
            )
            
            return {
                "optimizer": g_optimizer,
                "lr_scheduler": scheduler,
            }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
            
        Returns:
            Training loss.
        """
        source_mel = batch["source_mel"]
        target_mel = batch["target_mel"]
        
        # Forward pass
        outputs = self.model(source_mel, target_mel)
        
        # Compute losses
        losses = self.model.compute_losses(source_mel, target_mel, outputs)
        
        # Log losses
        for key, value in losses.items():
            self.log(f"train/{key}", value, on_step=True, on_epoch=True, prog_bar=True)
        
        # Return generator loss for optimization
        if "g_loss" in losses:
            return losses["g_loss"]
        else:
            return losses["total_loss"]
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Validation step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        """
        source_mel = batch["source_mel"]
        target_mel = batch["target_mel"]
        
        # Forward pass
        outputs = self.model(source_mel, target_mel)
        
        # Compute losses
        losses = self.model.compute_losses(source_mel, target_mel, outputs)
        
        # Log losses
        for key, value in losses.items():
            self.log(f"val/{key}", value, on_step=False, on_epoch=True, prog_bar=True)
        
        # Generate samples for visualization
        if batch_idx == 0:
            self._log_samples(source_mel, target_mel, outputs)
    
    def _log_samples(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Log sample spectrograms for visualization.
        
        Args:
            source_mel: Source mel spectrogram.
            target_mel: Target mel spectrogram.
            outputs: Model outputs.
        """
        # Take first sample from batch
        source_sample = source_mel[0].cpu()
        target_sample = target_mel[0].cpu()
        
        if "source_to_target" in outputs:
            converted_sample = outputs["source_to_target"][0].cpu()
        else:
            converted_sample = outputs["converted_mel"][0].cpu()
        
        # Log spectrograms
        self.logger.experiment.add_image(
            "source_mel", source_sample, self.current_epoch
        )
        self.logger.experiment.add_image(
            "target_mel", target_sample, self.current_epoch
        )
        self.logger.experiment.add_image(
            "converted_mel", converted_sample, self.current_epoch
        )
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Test step.
        
        Args:
            batch: Batch of data.
            batch_idx: Batch index.
        """
        source_mel = batch["source_mel"]
        target_mel = batch["target_mel"]
        
        # Forward pass
        outputs = self.model(source_mel, target_mel)
        
        # Compute losses
        losses = self.model.compute_losses(source_mel, target_mel, outputs)
        
        # Log losses
        for key, value in losses.items():
            self.log(f"test/{key}", value, on_step=False, on_epoch=True)
        
        # Generate audio samples if vocoder is available
        if self.vocoder is not None and batch_idx < 5:  # Limit to first 5 batches
            self._generate_audio_samples(source_mel, target_mel, outputs, batch_idx)
    
    def _generate_audio_samples(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Generate audio samples for testing.
        
        Args:
            source_mel: Source mel spectrogram.
            target_mel: Target mel spectrogram.
            outputs: Model outputs.
            batch_idx: Batch index.
        """
        # Take first sample from batch
        source_sample = source_mel[0:1]
        target_sample = target_mel[0:1]
        
        if "source_to_target" in outputs:
            converted_sample = outputs["source_to_target"][0:1]
        else:
            converted_sample = outputs["converted_mel"][0:1]
        
        # Generate audio
        with torch.no_grad():
            source_audio = self.vocoder(source_sample)
            target_audio = self.vocoder(target_sample)
            converted_audio = self.vocoder(converted_sample)
        
        # Save audio samples
        output_dir = Path("assets/test_samples")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy and save
        source_audio_np = source_audio[0].cpu().numpy()
        target_audio_np = target_audio[0].cpu().numpy()
        converted_audio_np = converted_audio[0].cpu().numpy()
        
        # Save as WAV files
        import soundfile as sf
        sf.write(output_dir / f"source_{batch_idx}.wav", source_audio_np, 22050)
        sf.write(output_dir / f"target_{batch_idx}.wav", target_audio_np, 22050)
        sf.write(output_dir / f"converted_{batch_idx}.wav", converted_audio_np, 22050)
    
    def on_train_epoch_end(self) -> None:
        """Called at the end of training epoch."""
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("train/lr", lr, on_epoch=True)
    
    def on_validation_epoch_end(self) -> None:
        """Called at the end of validation epoch."""
        # Save checkpoint
        if self.current_epoch % self.config.training.save_every_n_epochs == 0:
            checkpoint_path = Path("checkpoints") / f"epoch_{self.current_epoch}.ckpt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.trainer.save_checkpoint(checkpoint_path)


def train_model(
    model: Union[CycleGANVC, VAEVC],
    data_module: VoiceConversionDataModule,
    config: DictConfig,
    vocoder: Optional[HiFiGANVocoder] = None,
) -> pl.Trainer:
    """Train voice conversion model.
    
    Args:
        model: Voice conversion model.
        data_module: Data module.
        config: Training configuration.
        vocoder: Optional vocoder for audio generation.
        
    Returns:
        Trained model.
    """
    # Create trainer
    lightning_trainer = VoiceConversionTrainer(model, config, vocoder)
    
    # Create PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        log_every_n_steps=config.training.log_every_n_steps,
        val_check_interval=config.training.val_check_interval,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    
    # Train model
    trainer.fit(lightning_trainer, data_module)
    
    return trainer
