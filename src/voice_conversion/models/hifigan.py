"""HiFi-GAN vocoder for high-quality audio synthesis."""

import logging
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block for HiFi-GAN generator."""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        leaky_relu_slope: float = 0.1,
    ):
        """Initialize residual block.
        
        Args:
            channels: Number of channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation rate.
            leaky_relu_slope: LeakyReLU negative slope.
        """
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=dilation))
        self.leaky_relu = nn.LeakyReLU(leaky_relu_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        residual = x
        
        out = self.leaky_relu(x)
        out = self.conv1(out)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        
        return out + residual


class Generator(nn.Module):
    """HiFi-GAN generator."""
    
    def __init__(
        self,
        mel_channels: int = 80,
        audio_channels: int = 1,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    ):
        """Initialize HiFi-GAN generator.
        
        Args:
            mel_channels: Number of mel spectrogram channels.
            audio_channels: Number of audio channels.
            upsample_rates: Upsampling rates for each layer.
            upsample_kernel_sizes: Kernel sizes for upsampling layers.
            upsample_initial_channel: Initial number of channels.
            resblock_kernel_sizes: Kernel sizes for residual blocks.
            resblock_dilation_sizes: Dilation sizes for residual blocks.
        """
        super().__init__()
        
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        # Initial convolution
        self.conv_pre = weight_norm(nn.Conv1d(mel_channels, upsample_initial_channel, 7, padding=3))
        
        # Upsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(
                    upsample_initial_channel // (2 ** i),
                    upsample_initial_channel // (2 ** (i + 1)),
                    k, u, padding=(k - u) // 2
                )
            ))
        
        # Residual blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch, k, d[0]))
        
        # Final convolution
        self.conv_post = weight_norm(nn.Conv1d(ch, audio_channels, 7, padding=3))
        
        # Activation
        self.leaky_relu = nn.LeakyReLU(0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input mel spectrogram.
            
        Returns:
            Generated audio waveform.
        """
        # Initial convolution
        x = self.conv_pre(x)
        
        # Upsampling and residual blocks
        for i in range(self.num_upsamples):
            x = self.leaky_relu(x)
            x = self.ups[i](x)
            
            # Apply residual blocks
            for j in range(self.num_kernels):
                x = self.resblocks[i * self.num_kernels + j](x)
        
        # Final convolution
        x = self.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        
        return x
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for layer in self.modules():
            if hasattr(layer, 'weight_g'):
                nn.utils.remove_weight_norm(layer)


class Discriminator(nn.Module):
    """Multi-scale discriminator for HiFi-GAN."""
    
    def __init__(
        self,
        audio_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 7,
    ):
        """Initialize discriminator.
        
        Args:
            audio_channels: Number of audio channels.
            base_channels: Number of base channels.
            n_layers: Number of discriminator layers.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Convolution padding.
        """
        super().__init__()
        
        layers = []
        in_channels = audio_channels
        
        # Discriminator layers
        for i in range(n_layers):
            out_channels = base_channels * (2 ** i)
            layers.extend([
                weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)),
                nn.LeakyReLU(0.1),
            ])
            in_channels = out_channels
        
        # Final layer
        layers.append(weight_norm(nn.Conv1d(in_channels, 1, kernel_size, stride, padding)))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input audio waveform.
            
        Returns:
            Discriminator output.
        """
        return self.model(x)
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for layer in self.modules():
            if hasattr(layer, 'weight_g'):
                nn.utils.remove_weight_norm(layer)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator combining multiple discriminators."""
    
    def __init__(
        self,
        audio_channels: int = 1,
        base_channels: int = 64,
        n_layers: int = 3,
        kernel_sizes: List[int] = [15, 41, 5],
        strides: List[int] = [1, 2, 1],
        paddings: List[int] = [7, 20, 2],
    ):
        """Initialize multi-scale discriminator.
        
        Args:
            audio_channels: Number of audio channels.
            base_channels: Number of base channels.
            n_layers: Number of discriminator layers.
            kernel_sizes: Kernel sizes for each discriminator.
            strides: Strides for each discriminator.
            paddings: Paddings for each discriminator.
        """
        super().__init__()
        
        self.discriminators = nn.ModuleList([
            Discriminator(audio_channels, base_channels, n_layers, k, s, p)
            for k, s, p in zip(kernel_sizes, strides, paddings)
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input audio waveform.
            
        Returns:
            List of discriminator outputs.
        """
        outputs = []
        for discriminator in self.discriminators:
            outputs.append(discriminator(x))
        return outputs
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        for discriminator in self.discriminators:
            discriminator.remove_weight_norm()


class HiFiGANVocoder(nn.Module):
    """HiFi-GAN vocoder for high-quality audio synthesis."""
    
    def __init__(
        self,
        mel_channels: int = 80,
        audio_channels: int = 1,
        upsample_rates: List[int] = [8, 8, 2, 2],
        upsample_kernel_sizes: List[int] = [16, 16, 4, 4],
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: List[int] = [3, 7, 11],
        resblock_dilation_sizes: List[List[int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        lambda_adv: float = 1.0,
        lambda_fm: float = 2.0,
        lambda_mel: float = 45.0,
    ):
        """Initialize HiFi-GAN vocoder.
        
        Args:
            mel_channels: Number of mel spectrogram channels.
            audio_channels: Number of audio channels.
            upsample_rates: Upsampling rates for generator.
            upsample_kernel_sizes: Kernel sizes for upsampling layers.
            upsample_initial_channel: Initial number of channels.
            resblock_kernel_sizes: Kernel sizes for residual blocks.
            resblock_dilation_sizes: Dilation sizes for residual blocks.
            lambda_adv: Adversarial loss weight.
            lambda_fm: Feature matching loss weight.
            lambda_mel: Mel spectrogram loss weight.
        """
        super().__init__()
        
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        
        # Generator
        self.generator = Generator(
            mel_channels, audio_channels, upsample_rates, upsample_kernel_sizes,
            upsample_initial_channel, resblock_kernel_sizes, resblock_dilation_sizes
        )
        
        # Multi-scale discriminator
        self.discriminator = MultiScaleDiscriminator(audio_channels)
    
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            mel_spec: Input mel spectrogram.
            
        Returns:
            Generated audio waveform.
        """
        return self.generator(mel_spec)
    
    def compute_losses(
        self,
        mel_spec: torch.Tensor,
        audio: torch.Tensor,
        generated_audio: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute HiFi-GAN losses.
        
        Args:
            mel_spec: Input mel spectrogram.
            audio: Target audio waveform.
            generated_audio: Generated audio waveform.
            
        Returns:
            Dictionary containing loss values.
        """
        # Adversarial losses
        real_outputs = self.discriminator(audio)
        fake_outputs = self.discriminator(generated_audio.detach())
        
        d_loss = 0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            d_loss += F.mse_loss(real_out, torch.ones_like(real_out))
            d_loss += F.mse_loss(fake_out, torch.zeros_like(fake_out))
        
        # Generator losses
        fake_outputs = self.discriminator(generated_audio)
        
        g_loss = 0
        for fake_out in fake_outputs:
            g_loss += F.mse_loss(fake_out, torch.ones_like(fake_out))
        
        # Feature matching loss
        fm_loss = 0
        for real_out, fake_out in zip(real_outputs, fake_outputs):
            for real_feat, fake_feat in zip(real_out, fake_out):
                fm_loss += F.l1_loss(fake_feat, real_feat)
        
        # Mel spectrogram loss
        mel_loss = F.l1_loss(generated_audio, audio)
        
        # Total losses
        total_g_loss = self.lambda_adv * g_loss + self.lambda_fm * fm_loss + self.lambda_mel * mel_loss
        
        return {
            "g_loss": total_g_loss,
            "d_loss": d_loss,
            "fm_loss": fm_loss,
            "mel_loss": mel_loss,
        }
    
    def remove_weight_norm(self):
        """Remove weight normalization for inference."""
        self.generator.remove_weight_norm()
        self.discriminator.remove_weight_norm()
