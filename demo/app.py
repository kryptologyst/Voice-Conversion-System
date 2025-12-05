"""Streamlit demo for voice conversion."""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.voice_conversion import (
    CycleGANVC,
    VAEVC,
    AudioProcessor,
    set_seed,
    get_device,
    load_config,
)

# Page configuration
st.set_page_config(
    page_title="Voice Conversion Demo",
    page_icon="🎤",
    layout="wide",
)

# Title
st.title("🎤 Voice Conversion Demo")
st.markdown("Convert voice characteristics while preserving speech content using deep learning models.")

# Sidebar for model selection
st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["CycleGAN", "VAE"],
    help="Choose the voice conversion model architecture"
)

# Model parameters
if model_type == "CycleGAN":
    st.sidebar.subheader("CycleGAN Parameters")
    lambda_cycle = st.sidebar.slider("Cycle Loss Weight", 1.0, 20.0, 10.0)
    lambda_identity = st.sidebar.slider("Identity Loss Weight", 1.0, 10.0, 5.0)
    n_residual_blocks = st.sidebar.slider("Residual Blocks", 3, 9, 6)
else:
    st.sidebar.subheader("VAE Parameters")
    latent_dim = st.sidebar.slider("Latent Dimension", 64, 256, 128)
    speaker_dim = st.sidebar.slider("Speaker Dimension", 32, 128, 64)
    beta = st.sidebar.slider("Beta (KL Weight)", 0.1, 2.0, 1.0)

# Audio parameters
st.sidebar.subheader("Audio Parameters")
sample_rate = st.sidebar.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
n_mels = st.sidebar.slider("Mel Channels", 40, 128, 80)

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "processor" not in st.session_state:
    st.session_state.processor = None

# Model loading section
st.header("Model Loading")

col1, col2 = st.columns([2, 1])

with col1:
    model_path = st.file_uploader(
        "Upload Model Checkpoint",
        type=["pt", "pth", "ckpt"],
        help="Upload a trained voice conversion model checkpoint"
    )

with col2:
    if st.button("Load Model", disabled=model_path is None):
        with st.spinner("Loading model..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
                    tmp_file.write(model_path.read())
                    tmp_path = tmp_file.name
                
                # Create audio processor
                processor = AudioProcessor(
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                )
                
                # Create model based on type
                if model_type == "CycleGAN":
                    model = CycleGANVC(
                        lambda_cycle=lambda_cycle,
                        lambda_identity=lambda_identity,
                        n_residual_blocks=n_residual_blocks,
                    )
                else:
                    model = VAEVC(
                        latent_dim=latent_dim,
                        speaker_dim=speaker_dim,
                        beta=beta,
                    )
                
                # Load model weights
                device = get_device()
                checkpoint = torch.load(tmp_path, map_location=device)
                if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()
                
                # Store in session state
                st.session_state.model = model
                st.session_state.processor = processor
                
                st.success("Model loaded successfully!")
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")

# Voice conversion section
st.header("Voice Conversion")

if st.session_state.model is None:
    st.warning("Please load a model first.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Source Audio")
        source_audio = st.file_uploader(
            "Upload Source Audio",
            type=["wav", "mp3", "flac"],
            key="source",
            help="Upload the source voice audio file"
        )
        
        if source_audio:
            st.audio(source_audio, format="audio/wav")
    
    with col2:
        st.subheader("Target Audio")
        target_audio = st.file_uploader(
            "Upload Target Audio",
            type=["wav", "mp3", "flac"],
            key="target",
            help="Upload the target voice audio file"
        )
        
        if target_audio:
            st.audio(target_audio, format="audio/wav")
    
    # Conversion button
    if source_audio and target_audio:
        if st.button("Convert Voice", type="primary"):
            with st.spinner("Converting voice..."):
                try:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as source_tmp:
                        source_tmp.write(source_audio.read())
                        source_path = source_tmp.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as target_tmp:
                        target_tmp.write(target_audio.read())
                        target_path = target_tmp.name
                    
                    # Load audio files
                    processor = st.session_state.processor
                    model = st.session_state.model
                    device = get_device()
                    
                    source_audio_data = processor.load_audio(source_path)
                    target_audio_data = processor.load_audio(target_path)
                    
                    # Convert to mel spectrograms
                    source_mel = processor.mel_spectrogram(source_audio_data)
                    target_mel = processor.mel_spectrogram(target_audio_data)
                    
                    # Convert to tensors
                    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
                    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
                    
                    # Generate converted mel spectrogram
                    with torch.no_grad():
                        if hasattr(model, 'generate'):
                            # CycleGAN model
                            converted_mel = model.generate(source_mel_tensor)
                        else:
                            # VAE model
                            converted_mel = model.generate(source_mel_tensor, target_mel_tensor)
                    
                    # Convert back to numpy
                    converted_mel_np = converted_mel[0].cpu().numpy()
                    
                    # Convert mel spectrogram back to audio
                    converted_audio = processor.mel_to_audio(converted_mel_np)
                    
                    # Save converted audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_tmp:
                        processor.save_audio(converted_audio, output_tmp.name)
                        
                        # Read the converted audio for display
                        with open(output_tmp.name, "rb") as f:
                            converted_audio_bytes = f.read()
                    
                    st.success("Voice conversion completed!")
                    
                    # Display converted audio
                    st.subheader("Converted Audio")
                    st.audio(converted_audio_bytes, format="audio/wav")
                    
                    # Download button
                    st.download_button(
                        label="Download Converted Audio",
                        data=converted_audio_bytes,
                        file_name="converted_voice.wav",
                        mime="audio/wav"
                    )
                    
                    # Clean up temporary files
                    os.unlink(source_path)
                    os.unlink(target_path)
                    os.unlink(output_tmp.name)
                    
                except Exception as e:
                    st.error(f"Error during conversion: {str(e)}")

# Voice interpolation section
st.header("Voice Interpolation")

if st.session_state.model is None:
    st.warning("Please load a model first.")
else:
    st.markdown("Generate intermediate voices between source and target speakers.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        interp_source_audio = st.file_uploader(
            "Upload Source Audio for Interpolation",
            type=["wav", "mp3", "flac"],
            key="interp_source",
        )
        
        if interp_source_audio:
            st.audio(interp_source_audio, format="audio/wav")
    
    with col2:
        interp_target_audio = st.file_uploader(
            "Upload Target Audio for Interpolation",
            type=["wav", "mp3", "flac"],
            key="interp_target",
        )
        
        if interp_target_audio:
            st.audio(interp_target_audio, format="audio/wav")
    
    n_interpolations = st.slider("Number of Interpolation Steps", 3, 10, 5)
    
    if interp_source_audio and interp_target_audio:
        if st.button("Generate Interpolated Voices", type="primary"):
            with st.spinner("Generating interpolated voices..."):
                try:
                    # Save uploaded files temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as source_tmp:
                        source_tmp.write(interp_source_audio.read())
                        source_path = source_tmp.name
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as target_tmp:
                        target_tmp.write(interp_target_audio.read())
                        target_path = target_tmp.name
                    
                    # Load audio files
                    processor = st.session_state.processor
                    model = st.session_state.model
                    device = get_device()
                    
                    source_audio_data = processor.load_audio(source_path)
                    target_audio_data = processor.load_audio(target_path)
                    
                    # Convert to mel spectrograms
                    source_mel = processor.mel_spectrogram(source_audio_data)
                    target_mel = processor.mel_spectrogram(target_audio_data)
                    
                    # Convert to tensors
                    source_mel_tensor = torch.from_numpy(source_mel).float().unsqueeze(0).to(device)
                    target_mel_tensor = torch.from_numpy(target_mel).float().unsqueeze(0).to(device)
                    
                    # Create interpolation weights
                    alphas = np.linspace(0, 1, n_interpolations)
                    
                    st.subheader("Interpolated Voices")
                    
                    # Generate interpolated voices
                    for i, alpha in enumerate(alphas):
                        with torch.no_grad():
                            if hasattr(model, 'generate'):
                                # CycleGAN model - simple interpolation
                                converted_mel = model.generate(source_mel_tensor)
                                interpolated_mel = (1 - alpha) * source_mel_tensor + alpha * converted_mel
                            else:
                                # VAE model - interpolate in latent space
                                converted_mel = model.generate(source_mel_tensor, target_mel_tensor)
                                interpolated_mel = (1 - alpha) * source_mel_tensor + alpha * converted_mel
                        
                        # Convert back to numpy
                        interpolated_mel_np = interpolated_mel[0].cpu().numpy()
                        
                        # Convert mel spectrogram back to audio
                        interpolated_audio = processor.mel_to_audio(interpolated_mel_np)
                        
                        # Save interpolated audio
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as output_tmp:
                            processor.save_audio(interpolated_audio, output_tmp.name)
                            
                            # Read the interpolated audio for display
                            with open(output_tmp.name, "rb") as f:
                                interpolated_audio_bytes = f.read()
                        
                        # Display interpolated audio
                        st.write(f"Step {i+1} (α={alpha:.2f})")
                        st.audio(interpolated_audio_bytes, format="audio/wav")
                        
                        # Clean up temporary file
                        os.unlink(output_tmp.name)
                    
                    st.success("Voice interpolation completed!")
                    
                    # Clean up temporary files
                    os.unlink(source_path)
                    os.unlink(target_path)
                    
                except Exception as e:
                    st.error(f"Error during interpolation: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch for voice conversion research.")
