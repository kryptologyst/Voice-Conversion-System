"""Data loading and preprocessing for voice conversion."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, Spectrogram
import pytorch_lightning as pl

from .utils.audio import AudioProcessor

logger = logging.getLogger(__name__)


class VoiceConversionDataset(Dataset):
    """Dataset for voice conversion training."""
    
    def __init__(
        self,
        source_dir: Union[str, Path],
        target_dir: Union[str, Path],
        processor: AudioProcessor,
        max_length: Optional[int] = None,
        augment: bool = True,
    ):
        """Initialize voice conversion dataset.
        
        Args:
            source_dir: Directory containing source audio files.
            target_dir: Directory containing target audio files.
            processor: Audio processor instance.
            max_length: Maximum audio length in samples.
            augment: Whether to apply data augmentation.
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        
        # Get file lists
        self.source_files = sorted(list(self.source_dir.glob("*.wav")))
        self.target_files = sorted(list(self.target_dir.glob("*.wav")))
        
        if len(self.source_files) != len(self.target_files):
            raise ValueError(
                f"Mismatch in number of files: {len(self.source_files)} source, "
                f"{len(self.target_files)} target"
            )
        
        logger.info(f"Loaded {len(self.source_files)} audio pairs")
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.source_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset.
        
        Args:
            idx: Sample index.
            
        Returns:
            Dictionary containing source and target audio tensors.
        """
        # Load audio files
        source_audio = self.processor.load_audio(self.source_files[idx])
        target_audio = self.processor.load_audio(self.target_files[idx])
        
        # Trim silence
        source_audio = self.processor.trim_silence(source_audio)
        target_audio = self.processor.trim_silence(target_audio)
        
        # Pad or truncate to max_length
        if self.max_length:
            source_audio = self.processor.pad_audio(source_audio, self.max_length)
            target_audio = self.processor.pad_audio(target_audio, self.max_length)
        
        # Apply augmentation
        if self.augment:
            source_audio, target_audio = self._augment_pair(source_audio, target_audio)
        
        # Convert to mel spectrograms
        source_mel = self.processor.mel_spectrogram(source_audio)
        target_mel = self.processor.mel_spectrogram(target_audio)
        
        # Convert to tensors
        source_mel = torch.from_numpy(source_mel).float()
        target_mel = torch.from_numpy(target_mel).float()
        
        return {
            "source_mel": source_mel,
            "target_mel": target_mel,
            "source_audio": torch.from_numpy(source_audio).float(),
            "target_audio": torch.from_numpy(target_audio).float(),
        }
    
    def _augment_pair(self, source_audio: np.ndarray, target_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to audio pair.
        
        Args:
            source_audio: Source audio waveform.
            target_audio: Target audio waveform.
            
        Returns:
            Augmented audio pair.
        """
        # Random time stretching (synchronized)
        if np.random.random() < 0.3:
            stretch_factor = np.random.uniform(0.9, 1.1)
            min_length = min(len(source_audio), len(target_audio))
            new_length = int(min_length * stretch_factor)
            
            source_indices = np.linspace(0, min_length - 1, new_length)
            target_indices = np.linspace(0, min_length - 1, new_length)
            
            source_audio = np.interp(source_indices, np.arange(min_length), source_audio[:min_length])
            target_audio = np.interp(target_indices, np.arange(min_length), target_audio[:min_length])
        
        # Random pitch shifting (synchronized)
        if np.random.random() < 0.3:
            pitch_shift = np.random.uniform(-2, 2)  # semitones
            # Simple pitch shift by resampling
            shift_factor = 2 ** (pitch_shift / 12)
            new_length = int(len(source_audio) * shift_factor)
            
            if shift_factor > 1:
                # Upsample
                source_indices = np.linspace(0, len(source_audio) - 1, new_length)
                target_indices = np.linspace(0, len(target_audio) - 1, new_length)
                source_audio = np.interp(source_indices, np.arange(len(source_audio)), source_audio)
                target_audio = np.interp(target_indices, np.arange(len(target_audio)), target_audio)
            else:
                # Downsample
                indices = np.linspace(0, len(source_audio) - 1, new_length)
                source_audio = source_audio[indices.astype(int)]
                target_audio = target_audio[indices.astype(int)]
        
        # Random noise
        if np.random.random() < 0.2:
            noise_level = np.random.uniform(0.01, 0.05)
            source_audio += np.random.normal(0, noise_level, len(source_audio))
            target_audio += np.random.normal(0, noise_level, len(target_audio))
        
        return source_audio, target_audio


class VoiceConversionDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for voice conversion."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        processor: AudioProcessor,
        batch_size: int = 16,
        num_workers: int = 4,
        max_length: Optional[int] = None,
        train_split: float = 0.8,
        val_split: float = 0.1,
        augment: bool = True,
    ):
        """Initialize data module.
        
        Args:
            data_dir: Directory containing source and target subdirectories.
            processor: Audio processor instance.
            batch_size: Batch size for training.
            num_workers: Number of data loading workers.
            max_length: Maximum audio length in samples.
            train_split: Fraction of data for training.
            val_split: Fraction of data for validation.
            augment: Whether to apply data augmentation.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.augment = augment
        
        self.source_dir = self.data_dir / "source"
        self.target_dir = self.data_dir / "target"
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation/testing."""
        if stage == "fit" or stage is None:
            # Get all file indices
            source_files = sorted(list(self.source_dir.glob("*.wav")))
            n_files = len(source_files)
            
            # Split indices
            train_end = int(n_files * self.train_split)
            val_end = int(n_files * (self.train_split + self.val_split))
            
            train_indices = list(range(train_end))
            val_indices = list(range(train_end, val_end))
            
            # Create datasets
            self.train_dataset = self._create_dataset(train_indices, augment=True)
            self.val_dataset = self._create_dataset(val_indices, augment=False)
        
        if stage == "test" or stage is None:
            # Use remaining files for testing
            source_files = sorted(list(self.source_dir.glob("*.wav")))
            n_files = len(source_files)
            test_start = int(n_files * (self.train_split + self.val_split))
            test_indices = list(range(test_start, n_files))
            
            self.test_dataset = self._create_dataset(test_indices, augment=False)
    
    def _create_dataset(self, indices: List[int], augment: bool) -> VoiceConversionDataset:
        """Create dataset with specific file indices.
        
        Args:
            indices: List of file indices to include.
            augment: Whether to apply augmentation.
            
        Returns:
            VoiceConversionDataset instance.
        """
        # Create temporary directories with selected files
        temp_source = self.source_dir.parent / f"temp_source_{id(self)}"
        temp_target = self.data_dir.parent / f"temp_target_{id(self)}"
        
        temp_source.mkdir(exist_ok=True)
        temp_target.mkdir(exist_ok=True)
        
        source_files = sorted(list(self.source_dir.glob("*.wav")))
        target_files = sorted(list(self.target_dir.glob("*.wav")))
        
        for idx in indices:
            source_files[idx].symlink_to(temp_source / source_files[idx].name)
            target_files[idx].symlink_to(temp_target / target_files[idx].name)
        
        return VoiceConversionDataset(
            temp_source,
            temp_target,
            self.processor,
            self.max_length,
            augment,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
