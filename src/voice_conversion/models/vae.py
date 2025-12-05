"""VAE-based voice conversion model."""

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Encoder network for VAE voice conversion."""
    
    def __init__(
        self,
        input_channels: int = 1,
        latent_dim: int = 128,
        base_channels: int = 64,
        n_layers: int = 4,
    ):
        """Initialize encoder.
        
        Args:
            input_channels: Number of input channels.
            latent_dim: Dimension of latent space.
            base_channels: Number of base channels.
            n_layers: Number of encoder layers.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        
        layers = []
        in_channels = input_channels
        
        # Encoder layers
        for i in range(n_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space projection
        self.fc_mu = nn.Linear(base_channels * (2 ** (n_layers - 1)) * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * (2 ** (n_layers - 1)) * 4 * 4, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram.
            
        Returns:
            Mean and log variance of latent distribution.
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Encode
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        # Project to latent space
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """Decoder network for VAE voice conversion."""
    
    def __init__(
        self,
        latent_dim: int = 128,
        output_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 4,
    ):
        """Initialize decoder.
        
        Args:
            latent_dim: Dimension of latent space.
            output_channels: Number of output channels.
            base_channels: Number of base channels.
            n_layers: Number of decoder layers.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.n_layers = n_layers
        
        # Project from latent space
        self.fc = nn.Linear(latent_dim, base_channels * (2 ** (n_layers - 1)) * 4 * 4)
        
        layers = []
        in_channels = base_channels * (2 ** (n_layers - 1))
        
        # Decoder layers
        for i in range(n_layers - 1):
            out_channels = base_channels * (2 ** (n_layers - 2 - i))
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        # Final layer
        layers.append(nn.ConvTranspose2d(in_channels, output_channels, 4, stride=2, padding=1))
        layers.append(nn.Tanh())
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            z: Latent vector.
            
        Returns:
            Generated mel spectrogram.
        """
        # Project from latent space
        h = self.fc(z)
        h = h.view(h.size(0), self.base_channels * (2 ** (self.n_layers - 1)), 4, 4)
        
        # Decode
        return self.decoder(h)


class SpeakerEncoder(nn.Module):
    """Speaker encoder for voice conversion."""
    
    def __init__(
        self,
        input_channels: int = 1,
        speaker_dim: int = 64,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        """Initialize speaker encoder.
        
        Args:
            input_channels: Number of input channels.
            speaker_dim: Dimension of speaker embedding.
            base_channels: Number of base channels.
            n_layers: Number of encoder layers.
        """
        super().__init__()
        
        layers = []
        in_channels = input_channels
        
        # Encoder layers
        for i in range(n_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ])
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Speaker embedding projection
        self.fc_speaker = nn.Linear(base_channels * (2 ** (n_layers - 1)), speaker_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram.
            
        Returns:
            Speaker embedding.
        """
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Encode
        h = self.encoder(x)
        
        # Global average pooling
        h = self.global_pool(h)
        h = h.view(h.size(0), -1)
        
        # Project to speaker space
        speaker_embed = self.fc_speaker(h)
        
        return speaker_embed


class VAEVC(nn.Module):
    """VAE-based voice conversion model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        latent_dim: int = 128,
        speaker_dim: int = 64,
        base_channels: int = 64,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 4,
        beta: float = 1.0,
    ):
        """Initialize VAE voice conversion model.
        
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            latent_dim: Dimension of latent space.
            speaker_dim: Dimension of speaker embedding.
            base_channels: Number of base channels.
            n_encoder_layers: Number of encoder layers.
            n_decoder_layers: Number of decoder layers.
            beta: Beta-VAE parameter.
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.speaker_dim = speaker_dim
        self.beta = beta
        
        # Content encoder (VAE)
        self.content_encoder = Encoder(
            input_channels, latent_dim, base_channels, n_encoder_layers
        )
        self.content_decoder = Decoder(
            latent_dim, output_channels, base_channels, n_decoder_layers
        )
        
        # Speaker encoder
        self.speaker_encoder = SpeakerEncoder(
            input_channels, speaker_dim, base_channels, n_encoder_layers
        )
        
        # Speaker conditioning layer
        self.speaker_conditioning = nn.Linear(speaker_dim, latent_dim)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.
            
        Returns:
            Sampled latent vector.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            source_mel: Source mel spectrogram.
            target_mel: Target mel spectrogram.
            
        Returns:
            Dictionary containing model outputs.
        """
        # Encode content from source
        source_mu, source_logvar = self.content_encoder(source_mel)
        source_z = self.reparameterize(source_mu, source_logvar)
        
        # Encode speaker from target
        target_speaker = self.speaker_encoder(target_mel)
        
        # Condition content on target speaker
        speaker_condition = self.speaker_conditioning(target_speaker)
        conditioned_z = source_z + speaker_condition
        
        # Decode conditioned content
        converted_mel = self.content_decoder(conditioned_z)
        
        # Reconstruction for training
        target_mu, target_logvar = self.content_encoder(target_mel)
        target_z = self.reparameterize(target_mu, target_logvar)
        reconstructed_mel = self.content_decoder(target_z)
        
        return {
            "converted_mel": converted_mel,
            "reconstructed_mel": reconstructed_mel,
            "source_mu": source_mu,
            "source_logvar": source_logvar,
            "target_mu": target_mu,
            "target_logvar": target_logvar,
            "source_z": source_z,
            "target_z": target_z,
            "target_speaker": target_speaker,
        }
    
    def generate(self, source_mel: torch.Tensor, target_speaker_mel: torch.Tensor) -> torch.Tensor:
        """Generate converted voice.
        
        Args:
            source_mel: Source mel spectrogram.
            target_speaker_mel: Target speaker mel spectrogram.
            
        Returns:
            Generated converted mel spectrogram.
        """
        with torch.no_grad():
            # Encode content from source
            source_mu, source_logvar = self.content_encoder(source_mel)
            source_z = self.reparameterize(source_mu, source_logvar)
            
            # Encode speaker from target
            target_speaker = self.speaker_encoder(target_speaker_mel)
            
            # Condition content on target speaker
            speaker_condition = self.speaker_conditioning(target_speaker)
            conditioned_z = source_z + speaker_condition
            
            # Decode conditioned content
            converted_mel = self.content_decoder(conditioned_z)
            
            return converted_mel
    
    def compute_losses(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute VAE losses.
        
        Args:
            source_mel: Source mel spectrogram.
            target_mel: Target mel spectrogram.
            outputs: Model outputs.
            
        Returns:
            Dictionary containing loss values.
        """
        # Reconstruction loss
        recon_loss = F.l1_loss(outputs["reconstructed_mel"], target_mel)
        
        # KL divergence losses
        kl_source = -0.5 * torch.sum(
            1 + outputs["source_logvar"] - outputs["source_mu"].pow(2) - outputs["source_logvar"].exp()
        )
        kl_target = -0.5 * torch.sum(
            1 + outputs["target_logvar"] - outputs["target_mu"].pow(2) - outputs["target_logvar"].exp()
        )
        
        # Total loss
        total_loss = recon_loss + self.beta * (kl_source + kl_target)
        
        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_source": kl_source,
            "kl_target": kl_target,
        }
