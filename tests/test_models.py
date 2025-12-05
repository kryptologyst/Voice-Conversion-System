"""Unit tests for voice conversion models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.voice_conversion.models import CycleGANVC, VAEVC, HiFiGANVocoder
from src.voice_conversion.utils.audio import AudioProcessor
from src.voice_conversion.utils import set_seed, get_device


class TestCycleGANVC:
    """Test cases for CycleGAN voice conversion model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.model = CycleGANVC().to(self.device)
        
    def test_model_creation(self):
        """Test model creation."""
        assert isinstance(self.model, CycleGANVC)
        assert hasattr(self.model, 'G_source_to_target')
        assert hasattr(self.model, 'G_target_to_source')
        assert hasattr(self.model, 'D_source')
        assert hasattr(self.model, 'D_target')
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        target_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        outputs = self.model(source_mel, target_mel)
        
        # Check output keys
        expected_keys = [
            "source_to_target", "target_to_source",
            "source_to_target_to_source", "target_to_source_to_target",
            "source_identity", "target_identity",
            "d_source_real", "d_source_fake",
            "d_target_real", "d_target_fake"
        ]
        
        for key in expected_keys:
            assert key in outputs
            assert isinstance(outputs[key], torch.Tensor)
    
    def test_generate(self):
        """Test generation method."""
        batch_size = 1
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        with torch.no_grad():
            converted_mel = self.model.generate(source_mel)
        
        assert converted_mel.shape == source_mel.shape
        assert isinstance(converted_mel, torch.Tensor)
    
    def test_compute_losses(self):
        """Test loss computation."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        target_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        outputs = self.model(source_mel, target_mel)
        losses = self.model.compute_losses(source_mel, target_mel, outputs)
        
        # Check loss keys
        expected_loss_keys = [
            "g_loss", "d_loss", "cycle_loss_source", "cycle_loss_target",
            "identity_loss_source", "identity_loss_target"
        ]
        
        for key in expected_loss_keys:
            assert key in losses
            assert isinstance(losses[key], torch.Tensor)
            assert losses[key].item() >= 0  # Losses should be non-negative


class TestVAEVC:
    """Test cases for VAE voice conversion model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.model = VAEVC().to(self.device)
        
    def test_model_creation(self):
        """Test model creation."""
        assert isinstance(self.model, VAEVC)
        assert hasattr(self.model, 'content_encoder')
        assert hasattr(self.model, 'content_decoder')
        assert hasattr(self.model, 'speaker_encoder')
        assert hasattr(self.model, 'speaker_conditioning')
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        target_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        outputs = self.model(source_mel, target_mel)
        
        # Check output keys
        expected_keys = [
            "converted_mel", "reconstructed_mel",
            "source_mu", "source_logvar", "target_mu", "target_logvar",
            "source_z", "target_z", "target_speaker"
        ]
        
        for key in expected_keys:
            assert key in outputs
            assert isinstance(outputs[key], torch.Tensor)
    
    def test_generate(self):
        """Test generation method."""
        batch_size = 1
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        target_speaker_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        with torch.no_grad():
            converted_mel = self.model.generate(source_mel, target_speaker_mel)
        
        assert converted_mel.shape == source_mel.shape
        assert isinstance(converted_mel, torch.Tensor)
    
    def test_reparameterize(self):
        """Test reparameterization trick."""
        batch_size = 2
        latent_dim = 128
        
        mu = torch.randn(batch_size, latent_dim).to(self.device)
        logvar = torch.randn(batch_size, latent_dim).to(self.device)
        
        z = self.model.reparameterize(mu, logvar)
        
        assert z.shape == mu.shape
        assert isinstance(z, torch.Tensor)
    
    def test_compute_losses(self):
        """Test loss computation."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        
        source_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        target_mel = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        outputs = self.model(source_mel, target_mel)
        losses = self.model.compute_losses(source_mel, target_mel, outputs)
        
        # Check loss keys
        expected_loss_keys = [
            "total_loss", "recon_loss", "kl_source", "kl_target"
        ]
        
        for key in expected_loss_keys:
            assert key in losses
            assert isinstance(losses[key], torch.Tensor)
            assert losses[key].item() >= 0  # Losses should be non-negative


class TestHiFiGANVocoder:
    """Test cases for HiFi-GAN vocoder."""
    
    def setup_method(self):
        """Setup test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.model = HiFiGANVocoder().to(self.device)
        
    def test_model_creation(self):
        """Test model creation."""
        assert isinstance(self.model, HiFiGANVocoder)
        assert hasattr(self.model, 'generator')
        assert hasattr(self.model, 'discriminator')
    
    def test_forward_pass(self):
        """Test forward pass."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        
        mel_spec = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        
        with torch.no_grad():
            audio = self.model(mel_spec)
        
        assert isinstance(audio, torch.Tensor)
        assert audio.dim() == 3  # [batch, channels, time]
    
    def test_compute_losses(self):
        """Test loss computation."""
        batch_size = 2
        mel_channels = 80
        mel_length = 100
        audio_length = 1000
        
        mel_spec = torch.randn(batch_size, mel_channels, mel_length).to(self.device)
        audio = torch.randn(batch_size, 1, audio_length).to(self.device)
        
        with torch.no_grad():
            generated_audio = self.model(mel_spec)
        
        losses = self.model.compute_losses(mel_spec, audio, generated_audio)
        
        # Check loss keys
        expected_loss_keys = ["g_loss", "d_loss", "fm_loss", "mel_loss"]
        
        for key in expected_loss_keys:
            assert key in losses
            assert isinstance(losses[key], torch.Tensor)
            assert losses[key].item() >= 0  # Losses should be non-negative


class TestAudioProcessor:
    """Test cases for audio processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = AudioProcessor()
    
    def test_processor_creation(self):
        """Test processor creation."""
        assert isinstance(self.processor, AudioProcessor)
        assert hasattr(self.processor, 'sample_rate')
        assert hasattr(self.processor, 'n_mels')
        assert hasattr(self.processor, 'mel_filterbank')
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test normal audio
        audio = np.array([0.5, -0.3, 0.8, -0.1])
        normalized = self.processor.normalize_audio(audio)
        
        assert np.max(np.abs(normalized)) <= 1.0
        assert np.allclose(np.max(np.abs(normalized)), 1.0)
        
        # Test already normalized audio
        audio_normalized = np.array([0.1, -0.1, 0.05, -0.05])
        normalized_again = self.processor.normalize_audio(audio_normalized)
        
        assert np.allclose(normalized_again, audio_normalized)
    
    def test_pad_audio(self):
        """Test audio padding."""
        audio = np.array([1, 2, 3])
        target_length = 5
        
        padded = self.processor.pad_audio(audio, target_length)
        
        assert len(padded) == target_length
        assert np.array_equal(padded[:3], audio)
        assert np.all(padded[3:] == 0)
        
        # Test audio longer than target
        audio_long = np.array([1, 2, 3, 4, 5, 6])
        truncated = self.processor.pad_audio(audio_long, target_length)
        
        assert len(truncated) == target_length
        assert np.array_equal(truncated, audio_long[:target_length])
    
    def test_mel_spectrogram(self):
        """Test mel spectrogram computation."""
        # Generate test audio
        duration = 1.0
        sample_rate = 22050
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        mel_spec = self.processor.mel_spectrogram(audio)
        
        assert isinstance(mel_spec, np.ndarray)
        assert mel_spec.ndim == 2
        assert mel_spec.shape[0] == self.processor.n_mels
        assert mel_spec.shape[1] > 0  # Should have time dimension
    
    def test_mel_to_audio(self):
        """Test mel spectrogram to audio conversion."""
        # Generate test mel spectrogram
        n_mels = 80
        n_frames = 100
        mel_spec = np.random.randn(n_mels, n_frames)
        
        audio = self.processor.mel_to_audio(mel_spec)
        
        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1
        assert len(audio) > 0


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        np_rand1 = np.random.randn(10)
        torch_rand1 = torch.randn(10)
        
        # Set seed again
        set_seed(42)
        
        # Generate random numbers again
        np_rand2 = np.random.randn(10)
        torch_rand2 = torch.randn(10)
        
        # Should be the same
        assert np.allclose(np_rand1, np_rand2)
        assert torch.allclose(torch_rand1, torch_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cuda", "mps", "cpu"]


if __name__ == "__main__":
    pytest.main([__file__])
