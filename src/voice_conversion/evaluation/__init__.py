"""Evaluation metrics for voice conversion."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio
from torchmetrics.audio import SignalNoiseRatio, PerceptualEvaluationSpeechQuality
from scipy.stats import pearsonr
import librosa

logger = logging.getLogger(__name__)


class MelCepstralDistortion(Metric):
    """Mel Cepstral Distortion (MCD) metric for voice conversion evaluation."""
    
    def __init__(self, n_mfcc: int = 13, **kwargs):
        """Initialize MCD metric.
        
        Args:
            n_mfcc: Number of MFCC coefficients.
        """
        super().__init__(**kwargs)
        self.n_mfcc = n_mfcc
        self.add_state("mcd_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with new predictions and targets.
        
        Args:
            preds: Predicted mel spectrograms.
            target: Target mel spectrograms.
        """
        # Convert mel spectrograms to MFCCs
        pred_mfcc = self._mel_to_mfcc(preds)
        target_mfcc = self._mel_to_mfcc(target)
        
        # Compute MCD
        mcd = torch.mean(torch.sqrt(torch.sum((pred_mfcc - target_mfcc) ** 2, dim=1)))
        
        self.mcd_sum += mcd * preds.size(0)
        self.total_samples += preds.size(0)
    
    def compute(self) -> torch.Tensor:
        """Compute final MCD value."""
        return self.mcd_sum / self.total_samples
    
    def _mel_to_mfcc(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to MFCC.
        
        Args:
            mel_spec: Mel spectrogram.
            
        Returns:
            MFCC coefficients.
        """
        # Convert to numpy for librosa processing
        mel_np = mel_spec.cpu().numpy()
        mfcc_list = []
        
        for i in range(mel_np.shape[0]):
            # Convert mel to linear spectrogram
            linear_spec = librosa.feature.inverse.mel_to_stft(mel_np[i])
            
            # Convert to MFCC
            mfcc = librosa.feature.mfcc(S=linear_spec, n_mfcc=self.n_mfcc)
            mfcc_list.append(mfcc)
        
        return torch.from_numpy(np.array(mfcc_list)).to(mel_spec.device)


class SpectralConvergence(Metric):
    """Spectral Convergence metric for voice conversion evaluation."""
    
    def __init__(self, **kwargs):
        """Initialize Spectral Convergence metric."""
        super().__init__(**kwargs)
        self.add_state("sc_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_samples", default=torch.tensor(0), dist_reduce_fx="sum")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric with new predictions and targets.
        
        Args:
            preds: Predicted mel spectrograms.
            target: Target mel spectrograms.
        """
        # Compute spectral convergence
        sc = torch.norm(target - preds, p='fro') / torch.norm(target, p='fro')
        
        self.sc_sum += sc * preds.size(0)
        self.total_samples += preds.size(0)
    
    def compute(self) -> torch.Tensor:
        """Compute final Spectral Convergence value."""
        return self.sc_sum / self.total_samples


class VoiceConversionEvaluator:
    """Evaluator for voice conversion models."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        sample_rate: int = 22050,
    ):
        """Initialize evaluator.
        
        Args:
            device: Device to run evaluation on.
            sample_rate: Sample rate for audio processing.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        
        # Initialize metrics
        self.mcd = MelCepstralDistortion()
        self.sc = SpectralConvergence()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.sdr = SignalDistortionRatio()
        self.snr = SignalNoiseRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(self.sample_rate, "wb")
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocoder: Optional[torch.nn.Module] = None,
    ) -> Dict[str, float]:
        """Evaluate voice conversion model.
        
        Args:
            model: Voice conversion model.
            dataloader: Data loader for evaluation.
            vocoder: Optional vocoder for audio generation.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        model.eval()
        
        # Reset metrics
        self.mcd.reset()
        self.sc.reset()
        self.si_sdr.reset()
        self.sdr.reset()
        self.snr.reset()
        self.pesq.reset()
        
        all_metrics = {
            "mcd": [],
            "sc": [],
            "si_sdr": [],
            "sdr": [],
            "snr": [],
            "pesq": [],
        }
        
        with torch.no_grad():
            for batch in dataloader:
                source_mel = batch["source_mel"].to(self.device)
                target_mel = batch["target_mel"].to(self.device)
                source_audio = batch["source_audio"].to(self.device)
                target_audio = batch["target_audio"].to(self.device)
                
                # Generate converted mel spectrogram
                if hasattr(model, 'generate'):
                    converted_mel = model.generate(source_mel)
                else:
                    outputs = model(source_mel, target_mel)
                    converted_mel = outputs.get("converted_mel", outputs.get("source_to_target"))
                
                # Evaluate mel spectrogram metrics
                mcd_value = self.mcd(converted_mel, target_mel)
                sc_value = self.sc(converted_mel, target_mel)
                
                all_metrics["mcd"].append(mcd_value.item())
                all_metrics["sc"].append(sc_value.item())
                
                # Evaluate audio metrics if vocoder is available
                if vocoder is not None:
                    # Generate audio
                    converted_audio = vocoder(converted_mel)
                    
                    # Ensure same length
                    min_len = min(converted_audio.size(-1), target_audio.size(-1))
                    converted_audio = converted_audio[..., :min_len]
                    target_audio = target_audio[..., :min_len]
                    
                    # Compute audio metrics
                    si_sdr_value = self.si_sdr(converted_audio, target_audio)
                    sdr_value = self.sdr(converted_audio, target_audio)
                    snr_value = self.snr(converted_audio, target_audio)
                    pesq_value = self.pesq(converted_audio, target_audio)
                    
                    all_metrics["si_sdr"].append(si_sdr_value.item())
                    all_metrics["sdr"].append(sdr_value.item())
                    all_metrics["snr"].append(snr_value.item())
                    all_metrics["pesq"].append(pesq_value.item())
        
        # Compute final metrics
        final_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                final_metrics[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
        
        return final_metrics
    
    def evaluate_audio_quality(
        self,
        source_audio: np.ndarray,
        target_audio: np.ndarray,
        converted_audio: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate audio quality metrics.
        
        Args:
            source_audio: Source audio waveform.
            target_audio: Target audio waveform.
            converted_audio: Converted audio waveform.
            
        Returns:
            Dictionary containing audio quality metrics.
        """
        # Ensure same length
        min_len = min(len(source_audio), len(target_audio), len(converted_audio))
        source_audio = source_audio[:min_len]
        target_audio = target_audio[:min_len]
        converted_audio = converted_audio[:min_len]
        
        # Convert to torch tensors
        source_tensor = torch.from_numpy(source_audio).float().unsqueeze(0)
        target_tensor = torch.from_numpy(target_audio).float().unsqueeze(0)
        converted_tensor = torch.from_numpy(converted_audio).float().unsqueeze(0)
        
        # Compute metrics
        si_sdr = self.si_sdr(converted_tensor, target_tensor).item()
        sdr = self.sdr(converted_tensor, target_tensor).item()
        snr = self.snr(converted_tensor, target_tensor).item()
        pesq = self.pesq(converted_tensor, target_tensor).item()
        
        # Compute STOI (Short-Time Objective Intelligibility)
        stoi = self._compute_stoi(converted_audio, target_audio)
        
        # Compute FAD (Fréchet Audio Distance) - simplified version
        fad = self._compute_fad(converted_audio, target_audio)
        
        return {
            "si_sdr": si_sdr,
            "sdr": sdr,
            "snr": snr,
            "pesq": pesq,
            "stoi": stoi,
            "fad": fad,
        }
    
    def _compute_stoi(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute STOI metric.
        
        Args:
            pred: Predicted audio.
            target: Target audio.
            
        Returns:
            STOI value.
        """
        # Simplified STOI computation
        # In practice, you would use a proper STOI implementation
        try:
            from pystoi import stoi
            return stoi(target, pred, self.sample_rate, extended=False)
        except ImportError:
            logger.warning("pystoi not available, using simplified STOI")
            # Simplified correlation-based measure
            correlation = np.corrcoef(pred, target)[0, 1]
            return max(0, correlation)
    
    def _compute_fad(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute simplified FAD metric.
        
        Args:
            pred: Predicted audio.
            target: Target audio.
            
        Returns:
            FAD value.
        """
        # Simplified FAD computation using spectral features
        pred_spec = np.abs(librosa.stft(pred))
        target_spec = np.abs(librosa.stft(target))
        
        # Compute mean and covariance
        pred_mean = np.mean(pred_spec, axis=1)
        target_mean = np.mean(target_spec, axis=1)
        
        pred_cov = np.cov(pred_spec)
        target_cov = np.cov(target_spec)
        
        # Compute Fréchet distance
        mean_diff = np.sum((pred_mean - target_mean) ** 2)
        cov_diff = np.trace(pred_cov + target_cov - 2 * np.sqrt(pred_cov @ target_cov))
        
        return mean_diff + cov_diff
    
    def create_evaluation_report(
        self,
        metrics: Dict[str, Dict[str, float]],
        model_name: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Create evaluation report.
        
        Args:
            metrics: Evaluation metrics.
            model_name: Name of the model.
            output_path: Path to save report.
            
        Returns:
            Evaluation report as string.
        """
        report = f"Voice Conversion Evaluation Report\n"
        report += f"Model: {model_name}\n"
        report += f"{'='*50}\n\n"
        
        for metric_name, metric_values in metrics.items():
            report += f"{metric_name.upper()}:\n"
            report += f"  Mean: {metric_values['mean']:.4f}\n"
            report += f"  Std:  {metric_values['std']:.4f}\n"
            report += f"  Min:  {metric_values['min']:.4f}\n"
            report += f"  Max:  {metric_values['max']:.4f}\n\n"
        
        # Interpretation guidelines
        report += "Interpretation Guidelines:\n"
        report += "- MCD (Mel Cepstral Distortion): Lower is better (0-10 typical range)\n"
        report += "- SC (Spectral Convergence): Lower is better (0-1 range)\n"
        report += "- SI-SDR (Scale-Invariant SDR): Higher is better (dB)\n"
        report += "- SDR (Signal Distortion Ratio): Higher is better (dB)\n"
        report += "- SNR (Signal-to-Noise Ratio): Higher is better (dB)\n"
        report += "- PESQ (Perceptual Evaluation): Higher is better (1-5 range)\n"
        report += "- STOI (Short-Time Objective Intelligibility): Higher is better (0-1 range)\n"
        report += "- FAD (Fréchet Audio Distance): Lower is better\n"
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
        
        return report
