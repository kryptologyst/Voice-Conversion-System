"""CycleGAN-based voice conversion model."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """Residual block for generator networks."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        """Initialize residual block.
        
        Args:
            channels: Number of input/output channels.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout2d(dropout)
        self.norm1 = nn.InstanceNorm2d(channels)
        self.norm2 = nn.InstanceNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        residual = x
        
        out = self.norm1(self.conv1(x))
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.norm2(self.conv2(out))
        
        return out + residual


class Generator(nn.Module):
    """Generator network for voice conversion."""
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        n_residual_blocks: int = 6,
        dropout: float = 0.1,
    ):
        """Initialize generator.
        
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            base_channels: Number of base channels.
            n_residual_blocks: Number of residual blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        
        # Initial convolution
        self.initial = nn.Conv2d(input_channels, base_channels, 7, padding=3)
        
        # Downsampling
        self.down1 = nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels * 4, dropout=dropout)
            for _ in range(n_residual_blocks)
        ])
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 3, stride=2, padding=1, output_padding=1)
        
        # Final convolution
        self.final = nn.Conv2d(base_channels, output_channels, 7, padding=3)
        
        # Normalization layers
        self.norm_initial = nn.InstanceNorm2d(base_channels)
        self.norm_down1 = nn.InstanceNorm2d(base_channels * 2)
        self.norm_down2 = nn.InstanceNorm2d(base_channels * 4)
        self.norm_up1 = nn.InstanceNorm2d(base_channels * 2)
        self.norm_up2 = nn.InstanceNorm2d(base_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram.
            
        Returns:
            Generated mel spectrogram.
        """
        # Initial convolution
        out = self.norm_initial(self.initial(x))
        out = F.relu(out)
        
        # Downsampling
        out = self.norm_down1(self.down1(out))
        out = F.relu(out)
        
        out = self.norm_down2(self.down2(out))
        out = F.relu(out)
        
        # Residual blocks
        for residual_block in self.residual_blocks:
            out = residual_block(out)
        
        # Upsampling
        out = self.norm_up1(self.up1(out))
        out = F.relu(out)
        
        out = self.norm_up2(self.up2(out))
        out = F.relu(out)
        
        # Final convolution
        out = self.final(out)
        
        return torch.tanh(out)


class Discriminator(nn.Module):
    """Discriminator network for adversarial training."""
    
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
    ):
        """Initialize discriminator.
        
        Args:
            input_channels: Number of input channels.
            base_channels: Number of base channels.
            n_layers: Number of discriminator layers.
        """
        super().__init__()
        
        layers = []
        
        # First layer
        layers.append(spectral_norm(nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2))
        
        # Middle layers
        for i in range(n_layers - 1):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i + 1))
            layers.append(spectral_norm(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)))
            layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
        
        # Final layer
        final_channels = base_channels * (2 ** (n_layers - 1))
        layers.append(spectral_norm(nn.Conv2d(final_channels, 1, 4, stride=1, padding=1)))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram.
            
        Returns:
            Discriminator output.
        """
        return self.model(x)


class CycleGANVC(nn.Module):
    """CycleGAN-based voice conversion model."""
    
    def __init__(
        self,
        input_channels: int = 1,
        output_channels: int = 1,
        base_channels: int = 64,
        n_residual_blocks: int = 6,
        dropout: float = 0.1,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 5.0,
    ):
        """Initialize CycleGAN voice conversion model.
        
        Args:
            input_channels: Number of input channels.
            output_channels: Number of output channels.
            base_channels: Number of base channels.
            n_residual_blocks: Number of residual blocks.
            dropout: Dropout probability.
            lambda_cycle: Cycle consistency loss weight.
            lambda_identity: Identity loss weight.
        """
        super().__init__()
        
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        # Generators
        self.G_source_to_target = Generator(
            input_channels, output_channels, base_channels, n_residual_blocks, dropout
        )
        self.G_target_to_source = Generator(
            input_channels, output_channels, base_channels, n_residual_blocks, dropout
        )
        
        # Discriminators
        self.D_source = Discriminator(input_channels, base_channels)
        self.D_target = Discriminator(input_channels, base_channels)
    
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
            Dictionary containing generated spectrograms and discriminator outputs.
        """
        # Add channel dimension if needed
        if source_mel.dim() == 3:
            source_mel = source_mel.unsqueeze(1)
        if target_mel.dim() == 3:
            target_mel = target_mel.unsqueeze(1)
        
        # Generate converted spectrograms
        source_to_target = self.G_source_to_target(source_mel)
        target_to_source = self.G_target_to_source(target_mel)
        
        # Cycle consistency
        source_to_target_to_source = self.G_target_to_source(source_to_target)
        target_to_source_to_target = self.G_source_to_target(target_to_source)
        
        # Identity mapping
        source_identity = self.G_source_to_target(source_mel)
        target_identity = self.G_target_to_source(target_mel)
        
        # Discriminator outputs
        d_source_real = self.D_source(source_mel)
        d_source_fake = self.D_source(target_to_source)
        d_target_real = self.D_target(target_mel)
        d_target_fake = self.D_target(source_to_target)
        
        return {
            "source_to_target": source_to_target,
            "target_to_source": target_to_source,
            "source_to_target_to_source": source_to_target_to_source,
            "target_to_source_to_target": target_to_source_to_target,
            "source_identity": source_identity,
            "target_identity": target_identity,
            "d_source_real": d_source_real,
            "d_source_fake": d_source_fake,
            "d_target_real": d_target_real,
            "d_target_fake": d_target_fake,
        }
    
    def generate(self, source_mel: torch.Tensor) -> torch.Tensor:
        """Generate target voice from source voice.
        
        Args:
            source_mel: Source mel spectrogram.
            
        Returns:
            Generated target mel spectrogram.
        """
        if source_mel.dim() == 3:
            source_mel = source_mel.unsqueeze(1)
        
        with torch.no_grad():
            return self.G_source_to_target(source_mel)
    
    def compute_losses(
        self,
        source_mel: torch.Tensor,
        target_mel: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute CycleGAN losses.
        
        Args:
            source_mel: Source mel spectrogram.
            target_mel: Target mel spectrogram.
            outputs: Model outputs.
            
        Returns:
            Dictionary containing loss values.
        """
        # Adversarial losses
        g_loss_source_to_target = F.mse_loss(outputs["d_target_fake"], torch.ones_like(outputs["d_target_fake"]))
        g_loss_target_to_source = F.mse_loss(outputs["d_source_fake"], torch.ones_like(outputs["d_source_fake"]))
        
        d_loss_source_real = F.mse_loss(outputs["d_source_real"], torch.ones_like(outputs["d_source_real"]))
        d_loss_source_fake = F.mse_loss(outputs["d_source_fake"], torch.zeros_like(outputs["d_source_fake"]))
        d_loss_target_real = F.mse_loss(outputs["d_target_real"], torch.ones_like(outputs["d_target_real"]))
        d_loss_target_fake = F.mse_loss(outputs["d_target_fake"], torch.zeros_like(outputs["d_target_fake"]))
        
        # Cycle consistency losses
        cycle_loss_source = F.l1_loss(outputs["source_to_target_to_source"], source_mel)
        cycle_loss_target = F.l1_loss(outputs["target_to_source_to_target"], target_mel)
        
        # Identity losses
        identity_loss_source = F.l1_loss(outputs["source_identity"], source_mel)
        identity_loss_target = F.l1_loss(outputs["target_identity"], target_mel)
        
        # Total losses
        g_loss = (
            g_loss_source_to_target + g_loss_target_to_source +
            self.lambda_cycle * (cycle_loss_source + cycle_loss_target) +
            self.lambda_identity * (identity_loss_source + identity_loss_target)
        )
        
        d_loss = d_loss_source_real + d_loss_source_fake + d_loss_target_real + d_loss_target_fake
        
        return {
            "g_loss": g_loss,
            "d_loss": d_loss,
            "cycle_loss_source": cycle_loss_source,
            "cycle_loss_target": cycle_loss_target,
            "identity_loss_source": identity_loss_source,
            "identity_loss_target": identity_loss_target,
        }
