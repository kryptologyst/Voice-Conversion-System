"""Audio processing utilities for voice conversion."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Audio processing utilities for voice conversion."""
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        preemphasis: float = 0.97,
    ):
        """Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate.
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            n_mels: Number of mel filterbanks.
            f_min: Minimum frequency for mel filterbanks.
            f_max: Maximum frequency for mel filterbanks.
            preemphasis: Preemphasis coefficient.
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2
        self.preemphasis = preemphasis
        
        # Create mel filterbank
        self.mel_filterbank = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=self.f_max,
        )
    
    def load_audio(self, file_path: Union[str, Path]) -> np.ndarray:
        """Load audio file and resample to target sample rate.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Audio waveform as numpy array.
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def save_audio(self, audio: np.ndarray, file_path: Union[str, Path]) -> None:
        """Save audio to file.
        
        Args:
            audio: Audio waveform.
            file_path: Output file path.
        """
        sf.write(file_path, audio, self.sample_rate)
    
    def preemphasis_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply preemphasis filter to audio.
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Preemphasized audio.
        """
        return signal.lfilter([1, -self.preemphasis], [1], audio)
    
    def deemphasis_filter(self, audio: np.ndarray) -> np.ndarray:
        """Apply deemphasis filter to audio.
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Deemphasized audio.
        """
        return signal.lfilter([1], [1, -self.preemphasis], audio)
    
    def mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram from audio.
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Mel spectrogram.
        """
        # Apply preemphasis
        audio = self.preemphasis_filter(audio)
        
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window="hann",
        )
        
        # Convert to magnitude
        magnitude = np.abs(stft)
        
        # Apply mel filterbank
        mel_spec = np.dot(self.mel_filterbank, magnitude)
        
        # Convert to log scale
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec
    
    def mel_to_linear(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert mel spectrogram back to linear spectrogram.
        
        Args:
            mel_spec: Mel spectrogram.
            
        Returns:
            Linear spectrogram.
        """
        # Convert from log scale
        mel_spec = np.exp(mel_spec) - 1e-8
        
        # Apply inverse mel filterbank (pseudo-inverse)
        linear_spec = np.dot(np.linalg.pinv(self.mel_filterbank), mel_spec)
        
        return linear_spec
    
    def griffin_lim(self, magnitude_spec: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """Reconstruct audio from magnitude spectrogram using Griffin-Lim algorithm.
        
        Args:
            magnitude_spec: Magnitude spectrogram.
            n_iter: Number of iterations.
            
        Returns:
            Reconstructed audio waveform.
        """
        # Initialize random phase
        phase = np.random.random(magnitude_spec.shape).astype(np.complex64)
        
        for _ in range(n_iter):
            # Reconstruct complex spectrogram
            stft = magnitude_spec * np.exp(1j * np.angle(phase))
            
            # Convert back to time domain
            audio = librosa.istft(stft, hop_length=self.hop_length)
            
            # Convert back to frequency domain
            stft_new = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window="hann",
            )
            
            # Update phase
            phase = stft_new
        
        # Final reconstruction
        stft = magnitude_spec * np.exp(1j * np.angle(phase))
        audio = librosa.istft(stft, hop_length=self.hop_length)
        
        # Apply deemphasis
        audio = self.deemphasis_filter(audio)
        
        return audio
    
    def mel_to_audio(self, mel_spec: np.ndarray, n_iter: int = 60) -> np.ndarray:
        """Convert mel spectrogram back to audio.
        
        Args:
            mel_spec: Mel spectrogram.
            n_iter: Number of Griffin-Lim iterations.
            
        Returns:
            Reconstructed audio waveform.
        """
        # Convert mel to linear spectrogram
        linear_spec = self.mel_to_linear(mel_spec)
        
        # Reconstruct audio using Griffin-Lim
        audio = self.griffin_lim(linear_spec, n_iter)
        
        return audio
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio waveform.
            
        Returns:
            Normalized audio.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: float = 20) -> np.ndarray:
        """Trim silence from audio.
        
        Args:
            audio: Input audio waveform.
            top_db: Silence threshold in dB.
            
        Returns:
            Trimmed audio.
        """
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    def pad_audio(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad audio to target length.
        
        Args:
            audio: Input audio waveform.
            target_length: Target length in samples.
            
        Returns:
            Padded audio.
        """
        if len(audio) >= target_length:
            return audio[:target_length]
        
        pad_length = target_length - len(audio)
        return np.pad(audio, (0, pad_length), mode="constant")


def create_toy_dataset(
    output_dir: Union[str, Path],
    n_samples: int = 100,
    duration: float = 2.0,
    sample_rate: int = 22050,
) -> None:
    """Create a toy dataset for testing voice conversion.
    
    Args:
        output_dir: Output directory for the dataset.
        n_samples: Number of samples to generate.
        duration: Duration of each sample in seconds.
        sample_rate: Sample rate for generated audio.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = AudioProcessor(sample_rate=sample_rate)
    
    # Create source and target directories
    source_dir = output_dir / "source"
    target_dir = output_dir / "target"
    source_dir.mkdir(exist_ok=True)
    target_dir.mkdir(exist_ok=True)
    
    logger.info(f"Creating toy dataset with {n_samples} samples...")
    
    for i in range(n_samples):
        # Generate synthetic audio with different characteristics
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Source: lower frequency content
        source_freq = 200 + i * 10  # Varying frequency
        source_audio = np.sin(2 * np.pi * source_freq * t) * 0.5
        source_audio += np.sin(2 * np.pi * source_freq * 2 * t) * 0.3
        source_audio += np.random.normal(0, 0.1, len(t))  # Add noise
        
        # Target: higher frequency content
        target_freq = 300 + i * 15
        target_audio = np.sin(2 * np.pi * target_freq * t) * 0.5
        target_audio += np.sin(2 * np.pi * target_freq * 1.5 * t) * 0.3
        target_audio += np.random.normal(0, 0.1, len(t))  # Add noise
        
        # Normalize and save
        source_audio = processor.normalize_audio(source_audio)
        target_audio = processor.normalize_audio(target_audio)
        
        processor.save_audio(source_audio, source_dir / f"sample_{i:03d}.wav")
        processor.save_audio(target_audio, target_dir / f"sample_{i:03d}.wav")
    
    logger.info(f"Toy dataset created in {output_dir}")
