#!/usr/bin/env python3
"""Script to generate toy dataset for voice conversion training."""

import argparse
import logging
from pathlib import Path

from src.voice_conversion.utils.audio import create_toy_dataset

logger = logging.getLogger(__name__)


def main():
    """Main function to generate toy dataset."""
    parser = argparse.ArgumentParser(description="Generate toy dataset for voice conversion")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/toy_dataset",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of each sample in seconds",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Sample rate for generated audio",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating toy dataset in {output_dir}")
    logger.info(f"Parameters: {args.n_samples} samples, {args.duration}s duration, {args.sample_rate}Hz")
    
    # Generate dataset
    create_toy_dataset(
        output_dir=output_dir,
        n_samples=args.n_samples,
        duration=args.duration,
        sample_rate=args.sample_rate,
    )
    
    logger.info("Dataset generation completed!")


if __name__ == "__main__":
    main()
