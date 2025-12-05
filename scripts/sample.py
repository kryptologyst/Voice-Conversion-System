#!/usr/bin/env python3
"""Script for voice conversion inference and sampling."""

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf

from src.voice_conversion import (
    CycleGANVC,
    VAEVC,
    HiFiGANVocoder,
    AudioProcessor,
    set_seed,
    get_device,
    load_config,
)

logger = logging.getLogger(__name__)


def load_model(
    model_path: str,
    config: DictConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained voice conversion model.
    
    Args:
        model_path: Path to model checkpoint.
        config: Configuration object.
        device: Device to load model on.
        
    Returns:
        Loaded model.
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
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        # PyTorch Lightning checkpoint
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def convert_voice(
    source_audio_path: str,
    target_audio_path: str,
    model: torch.nn.Module,
    processor: AudioProcessor,
    device: torch.device,
    output_path: str,
) -> None:
    """Convert voice from source to target speaker.
    
    Args:
        source_audio_path: Path to source audio file.
        target_audio_path: Path to target audio file.
        model: Voice conversion model.
        processor: Audio processor.
        device: Device to run inference on.
        output_path: Path to save converted audio.
    """
    # Load audio files
    source_audio = processor.load_audio(source_audio_path)
    target_audio = processor.load_audio(target_audio_path)
    
    # Convert to mel spectrograms
    source_mel = processor.mel_spectrogram(source_audio)
    target_mel = processor.mel_spectrogram(target_audio)
    
    # Convert to tensors
    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
    
    # Generate converted mel spectrogram
    with torch.no_grad():
        if hasattr(model, 'generate'):
            # CycleGAN model
            converted_mel = model.generate(source_mel_tensor)
        else:
            # VAE model
            converted_mel = model.generate(source_mel_tensor, target_mel_tensor)
    
    # Convert back to numpy
    converted_mel_np = converted_mel[0].cpu().numpy()
    
    # Convert mel spectrogram back to audio
    converted_audio = processor.mel_to_audio(converted_mel_np)
    
    # Save converted audio
    processor.save_audio(converted_audio, output_path)
    
    logger.info(f"Converted audio saved to {output_path}")


def interpolate_voices(
    source_audio_path: str,
    target_audio_path: str,
    model: torch.nn.Module,
    processor: AudioProcessor,
    device: torch.device,
    output_dir: str,
    n_interpolations: int = 5,
) -> None:
    """Interpolate between source and target voices.
    
    Args:
        source_audio_path: Path to source audio file.
        target_audio_path: Path to target audio file.
        model: Voice conversion model.
        processor: Audio processor.
        device: Device to run inference on.
        output_dir: Directory to save interpolated audio.
        n_interpolations: Number of interpolation steps.
    """
    # Load audio files
    source_audio = processor.load_audio(source_audio_path)
    target_audio = processor.load_audio(target_audio_path)
    
    # Convert to mel spectrograms
    source_mel = processor.mel_spectrogram(source_audio)
    target_mel = processor.mel_spectrogram(target_audio)
    
    # Convert to tensors
    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
    
    # Create interpolation weights
    alphas = np.linspace(0, 1, n_interpolations)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, alpha in enumerate(alphas):
        with torch.no_grad():
            if hasattr(model, 'generate'):
                # CycleGAN model - simple interpolation
                converted_mel = model.generate(source_mel_tensor)
                interpolated_mel = (1 - alpha) * source_mel_tensor + alpha * converted_mel
            else:
                # VAE model - interpolate in latent space
                # This is a simplified interpolation
                converted_mel = model.generate(source_mel_tensor, target_mel_tensor)
                interpolated_mel = (1 - alpha) * source_mel_tensor + alpha * converted_mel
        
        # Convert back to numpy
        interpolated_mel_np = interpolated_mel[0].cpu().numpy()
        
        # Convert mel spectrogram back to audio
        interpolated_audio = processor.mel_to_audio(interpolated_mel_np)
        
        # Save interpolated audio
        output_path = output_dir / f"interpolation_{i:03d}_alpha_{alpha:.2f}.wav"
        processor.save_audio(interpolated_audio, output_path)
        
        logger.info(f"Interpolated audio saved to {output_path}")


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Voice conversion inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--source_audio",
        type=str,
        required=True,
        help="Path to source audio file",
    )
    parser.add_argument(
        "--target_audio",
        type=str,
        required=True,
        help="Path to target audio file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="converted_audio.wav",
        help="Path to save converted audio",
    )
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate interpolated voices",
    )
    parser.add_argument(
        "--n_interpolations",
        type=int,
        default=5,
        help="Number of interpolation steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config)
    
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
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, config, device)
    
    if args.interpolate:
        # Generate interpolated voices
        logger.info("Generating interpolated voices...")
        interpolate_voices(
            args.source_audio,
            args.target_audio,
            model,
            processor,
            device,
            Path(args.output_path).parent,
            args.n_interpolations,
        )
    else:
        # Convert voice
        logger.info("Converting voice...")
        convert_voice(
            args.source_audio,
            args.target_audio,
            model,
            processor,
            device,
            args.output_path,
        )
    
    logger.info("Inference completed!")


if __name__ == "__main__":
    main()
