#!/usr/bin/env python3
"""Example script demonstrating voice conversion system."""

import logging
import tempfile
from pathlib import Path

import torch
import numpy as np

from src.voice_conversion import (
    CycleGANVC,
    VAEVC,
    AudioProcessor,
    set_seed,
    get_device,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_audio(duration: float = 2.0, sample_rate: int = 22050) -> np.ndarray:
    """Create sample audio for demonstration.
    
    Args:
        duration: Duration in seconds.
        sample_rate: Sample rate.
        
    Returns:
        Sample audio waveform.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a simple tone with harmonics
    frequency = 220  # A3 note
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.3
    audio += np.sin(2 * np.pi * frequency * 3 * t) * 0.2
    
    # Add some noise
    audio += np.random.normal(0, 0.05, len(t))
    
    return audio


def demonstrate_cyclegan():
    """Demonstrate CycleGAN voice conversion."""
    logger.info("Demonstrating CycleGAN Voice Conversion")
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Create audio processor
    processor = AudioProcessor(sample_rate=22050, n_mels=80)
    
    # Create CycleGAN model
    model = CycleGANVC().to(device)
    model.eval()
    
    # Create sample audio
    source_audio = create_sample_audio()
    target_audio = create_sample_audio() * 0.8  # Slightly different amplitude
    
    # Convert to mel spectrograms
    source_mel = processor.mel_spectrogram(source_audio)
    target_mel = processor.mel_spectrogram(target_audio)
    
    # Convert to tensors
    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
    
    # Generate converted mel spectrogram
    with torch.no_grad():
        converted_mel = model.generate(source_mel_tensor)
    
    # Convert back to audio
    converted_mel_np = converted_mel[0].cpu().numpy()
    converted_audio = processor.mel_to_audio(converted_mel_np)
    
    # Save audio files
    output_dir = Path("assets/demo_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor.save_audio(source_audio, output_dir / "cyclegan_source.wav")
    processor.save_audio(target_audio, output_dir / "cyclegan_target.wav")
    processor.save_audio(converted_audio, output_dir / "cyclegan_converted.wav")
    
    logger.info(f"CycleGAN samples saved to {output_dir}")
    
    # Compute some basic metrics
    mse_loss = np.mean((converted_mel_np - target_mel) ** 2)
    logger.info(f"CycleGAN MSE Loss: {mse_loss:.4f}")


def demonstrate_vae():
    """Demonstrate VAE voice conversion."""
    logger.info("Demonstrating VAE Voice Conversion")
    
    # Set random seed
    set_seed(42)
    
    # Get device
    device = get_device()
    
    # Create audio processor
    processor = AudioProcessor(sample_rate=22050, n_mels=80)
    
    # Create VAE model
    model = VAEVC().to(device)
    model.eval()
    
    # Create sample audio
    source_audio = create_sample_audio()
    target_audio = create_sample_audio() * 0.8
    
    # Convert to mel spectrograms
    source_mel = processor.mel_spectrogram(source_audio)
    target_mel = processor.mel_spectrogram(target_audio)
    
    # Convert to tensors
    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
    
    # Generate converted mel spectrogram
    with torch.no_grad():
        converted_mel = model.generate(source_mel_tensor, target_mel_tensor)
    
    # Convert back to audio
    converted_mel_np = converted_mel[0].cpu().numpy()
    converted_audio = processor.mel_to_audio(converted_mel_np)
    
    # Save audio files
    output_dir = Path("assets/demo_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor.save_audio(source_audio, output_dir / "vae_source.wav")
    processor.save_audio(target_audio, output_dir / "vae_target.wav")
    processor.save_audio(converted_audio, output_dir / "vae_converted.wav")
    
    logger.info(f"VAE samples saved to {output_dir}")
    
    # Compute some basic metrics
    mse_loss = np.mean((converted_mel_np - target_mel) ** 2)
    logger.info(f"VAE MSE Loss: {mse_loss:.4f}")


def demonstrate_audio_processing():
    """Demonstrate audio processing capabilities."""
    logger.info("Demonstrating Audio Processing")
    
    # Create audio processor
    processor = AudioProcessor(sample_rate=22050, n_mels=80)
    
    # Create sample audio
    audio = create_sample_audio()
    
    # Test various processing functions
    normalized_audio = processor.normalize_audio(audio)
    padded_audio = processor.pad_audio(audio, len(audio) + 1000)
    trimmed_audio = processor.trim_silence(audio)
    
    # Convert to mel spectrogram and back
    mel_spec = processor.mel_spectrogram(audio)
    reconstructed_audio = processor.mel_to_audio(mel_spec)
    
    logger.info(f"Original audio length: {len(audio)}")
    logger.info(f"Normalized audio max: {np.max(np.abs(normalized_audio)):.4f}")
    logger.info(f"Padded audio length: {len(padded_audio)}")
    logger.info(f"Mel spectrogram shape: {mel_spec.shape}")
    logger.info(f"Reconstructed audio length: {len(reconstructed_audio)}")


def main():
    """Main demonstration function."""
    logger.info("Voice Conversion System Demonstration")
    logger.info("=" * 50)
    
    try:
        # Demonstrate audio processing
        demonstrate_audio_processing()
        logger.info("")
        
        # Demonstrate CycleGAN
        demonstrate_cyclegan()
        logger.info("")
        
        # Demonstrate VAE
        demonstrate_vae()
        logger.info("")
        
        logger.info("Demonstration completed successfully!")
        logger.info("Check the 'assets/demo_samples' directory for generated audio files.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
