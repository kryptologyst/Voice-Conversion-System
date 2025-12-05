#!/usr/bin/env python3
"""Setup script for voice conversion system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status.
    
    Args:
        command: Command to run.
        description: Description of the command.
        
    Returns:
        True if command succeeded, False otherwise.
    """
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"  Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("Voice Conversion System Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("Error: Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("Failed to install dependencies. Please check your Python environment.")
        sys.exit(1)
    
    # Create necessary directories
    directories = [
        "data/toy_dataset",
        "assets/demo_samples",
        "checkpoints",
        "logs",
        "outputs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Generate toy dataset
    if not run_command(
        "python scripts/generate_data.py --n_samples 50 --duration 1.0",
        "Generating toy dataset"
    ):
        print("Failed to generate toy dataset. Continuing anyway...")
    
    # Run demo
    print("\nRunning demonstration...")
    if run_command("python scripts/demo.py", "Running demonstration"):
        print("\n✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Train a model: python scripts/train.py --config configs/cyclegan.yaml")
        print("2. Run the demo: streamlit run demo/app.py")
        print("3. Check generated samples in assets/demo_samples/")
    else:
        print("\n✗ Setup completed with errors. Please check the logs above.")


if __name__ == "__main__":
    main()
