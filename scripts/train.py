#!/usr/bin/env python3
"""Training script for voice conversion models."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig, OmegaConf

from src.voice_conversion import (
    CycleGANVC,
    VAEVC,
    HiFiGANVocoder,
    AudioProcessor,
    VoiceConversionDataModule,
    VoiceConversionTrainer,
    VoiceConversionEvaluator,
    set_seed,
    get_device,
    load_config,
)

logger = logging.getLogger(__name__)


def setup_logging(config: DictConfig) -> None:
    """Setup logging configuration.
    
    Args:
        config: Configuration object.
    """
    log_level = getattr(logging, config.logging.level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(Path(config.logging.log_dir) / "training.log"),
            logging.StreamHandler(),
        ],
    )


def create_model(config: DictConfig) -> torch.nn.Module:
    """Create voice conversion model based on configuration.
    
    Args:
        config: Configuration object.
        
    Returns:
        Voice conversion model.
    """
    model_config = config.model
    
    if model_config.name == "cyclegan":
        model = CycleGANVC(
            input_channels=model_config.input_channels,
            output_channels=model_config.output_channels,
            base_channels=model_config.base_channels,
            n_residual_blocks=model_config.n_residual_blocks,
            dropout=model_config.dropout,
            lambda_cycle=model_config.lambda_cycle,
            lambda_identity=model_config.lambda_identity,
        )
    elif model_config.name == "vae":
        model = VAEVC(
            input_channels=model_config.input_channels,
            output_channels=model_config.output_channels,
            base_channels=model_config.base_channels,
            latent_dim=model_config.latent_dim,
            speaker_dim=model_config.speaker_dim,
            beta=model_config.beta,
        )
    else:
        raise ValueError(f"Unknown model: {model_config.name}")
    
    return model


def create_vocoder(config: DictConfig) -> Optional[HiFiGANVocoder]:
    """Create HiFi-GAN vocoder.
    
    Args:
        config: Configuration object.
        
    Returns:
        HiFi-GAN vocoder or None.
    """
    # For now, return None as vocoder training is complex
    # In practice, you would load a pre-trained vocoder
    return None


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train voice conversion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints and logs",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override data directory if provided
    if args.data_dir:
        config.data.data_dir = args.data_dir
    
    # Setup logging
    setup_logging(config)
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create audio processor
    audio_config = config.audio
    processor = AudioProcessor(
        sample_rate=audio_config.sample_rate,
        n_fft=audio_config.n_fft,
        hop_length=audio_config.hop_length,
        n_mels=audio_config.n_mels,
        f_min=audio_config.f_min,
        f_max=audio_config.f_max,
        preemphasis=audio_config.preemphasis,
    )
    
    # Create data module
    data_module = VoiceConversionDataModule(
        data_dir=config.data.data_dir,
        processor=processor,
        batch_size=config.training.batch_size,
        num_workers=config.data.num_workers,
        max_length=config.data.max_length,
        train_split=config.data.train_split,
        val_split=config.data.val_split,
        augment=config.data.augment,
    )
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create vocoder (optional)
    vocoder = create_vocoder(config)
    if vocoder:
        vocoder = vocoder.to(device)
    
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
        default_root_dir=args.output_dir,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(lightning_trainer, data_module, ckpt_path=args.resume)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = VoiceConversionEvaluator(device=device)
    test_metrics = evaluator.evaluate_model(model, data_module.test_dataloader(), vocoder)
    
    # Create evaluation report
    report = evaluator.create_evaluation_report(
        test_metrics,
        config.model.name,
        Path(args.output_dir) / "evaluation_report.txt",
    )
    
    logger.info("Evaluation Report:")
    logger.info(report)
    
    # Save final model
    final_model_path = Path(args.output_dir) / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


if __name__ == "__main__":
    main()
