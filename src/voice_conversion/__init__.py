"""Voice Conversion System - A modern deep learning approach to voice conversion."""

__version__ = "1.0.0"
__author__ = "AI Projects"

from .data import VoiceConversionDataset, VoiceConversionDataModule
from .models import CycleGANVC, VAEVC, HiFiGANVocoder
from .training import VoiceConversionTrainer
from .evaluation import VoiceConversionEvaluator
from .utils import set_seed, get_device, load_config

__all__ = [
    "VoiceConversionDataset",
    "VoiceConversionDataModule", 
    "CycleGANVC",
    "VAEVC",
    "HiFiGANVocoder",
    "VoiceConversionTrainer",
    "VoiceConversionEvaluator",
    "set_seed",
    "get_device",
    "load_config",
]
