#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import os

def analyze_audio_fourier(audio_file):
    """Analyze an audio file using Fourier transforms."""
    
    # Load audio file
    try:
        # Use librosa to load audio (handles many formats)
        y, sr = librosa.load(audio_file, sr=None)
        print(f"Loaded audio: {audio_file}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {len(y)/sr:.2f} seconds")
        print(f"Audio shape: {y.shape}")
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None
    
    # Create comprehensive analysis
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Time domain waveform
    plt.subplot(2, 3, 1)
    time = np.linspace(0, len(y)/sr, len(y))
    plt.plot(time, y)
    plt.title('Audio Waveform (Time Domain)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    
    # 2. FFT - Full frequency spectrum
    plt.subplot(2, 3, 2)
    fft = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), 1/sr)
    magnitude = np.abs(fft)
    
    # Only plot positive frequencies
    pos_mask = freqs > 0
    plt.loglog(freqs[pos_mask], magnitude[pos_mask])
    plt.title('Frequency Spectrum (Full FFT)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True, alpha=0.3)
    
    # 3. Spectrogram (time vs frequency)
    plt.subplot(2, 3, 3)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
    plt.title('Spectrogram (Time vs Frequency)')
    plt.colorbar(format='%+2.0f dB')
    
    # 4. Chromagram (musical notes)
    plt.subplot(2, 3, 4)
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', sr=sr)
    plt.title('Chromagram (Musical Notes)')
    plt.colorbar()
    
    # 5. Mel-frequency spectrogram
    plt.subplot(2, 3, 5)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.amplitude_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, y_axis='mel', x_axis='time', sr=sr)
    plt.title('Mel-frequency Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # 6. Dominant frequencies over time
    plt.subplot(2, 3, 6)
    # Short-time FFT to see how frequencies change over time
    hop_length = 512
    stft = librosa.stft(y, hop_length=hop_length)
    stft_magnitude = np.abs(stft)
    
    # Find dominant frequency in each time frame
    freqs_stft = librosa.fft_frequencies(sr=sr)
    times_stft = librosa.frames_to_time(np.arange(stft_magnitude.shape[1]), sr=sr, hop_length=hop_length)
    
    dominant_freq_idx = np.argmax(stft_magnitude, axis=0)
    dominant_freqs = freqs_stft[dominant_freq_idx]
    
    plt.plot(times_stft, dominant_freqs)
    plt.title('Dominant Frequency Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency (Hz)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the analysis
    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    output_file = f'audio_fourier_analysis_{base_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Analysis saved as: {output_file}")
    
    plt.show()
    
    # Print some interesting statistics
    print("\n" + "="*50)
    print("AUDIO ANALYSIS RESULTS")
    print("="*50)
    
    # Fundamental frequency estimation
    f0 = librosa.yin(y, fmin=80, fmax=400, sr=sr)
    f0_clean = f0[~np.isnan(f0)]
    if len(f0_clean) > 0:
        print(f"Estimated fundamental frequency: {np.mean(f0_clean):.1f} Hz")
    
    # Spectral centroid (brightness)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    print(f"Average spectral centroid: {np.mean(spectral_centroids):.1f} Hz")
    print("(Higher = brighter/more high-frequency content)")
    
    # Zero crossing rate (roughness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    print(f"Average zero crossing rate: {np.mean(zcr):.4f}")
    print("(Higher = more noisy/percussive)")
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    print(f"Average RMS energy: {np.mean(rms):.4f}")
    print("(Higher = louder)")
    
    # Dominant frequencies in the spectrum
    print(f"\nTop 5 dominant frequencies:")
    peak_indices = np.argsort(magnitude[pos_mask])[-5:][::-1]
    for i, idx in enumerate(peak_indices):
        freq = freqs[pos_mask][idx]
        mag = magnitude[pos_mask][idx]
        print(f"{i+1}. {freq:.1f} Hz (magnitude: {mag:.0f})")
    
    return y, sr, fft, freqs, magnitude

def generate_test_audio():
    """Generate a simple test audio file for demonstration."""
    duration = 3.0  # seconds
    sample_rate = 44100
    
    # Create time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a chord: C major (C, E, G) + some harmonics
    frequencies = [261.63, 329.63, 392.00, 523.25]  # C4, E4, G4, C5
    weights = [1.0, 0.8, 0.6, 0.3]
    
    audio = np.zeros_like(t)
    for freq, weight in zip(frequencies, weights):
        audio += weight * np.sin(2 * np.pi * freq * t)
    
    # Add some decay envelope
    envelope = np.exp(-t * 0.8)
    audio *= envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV file
    test_file = 'test_chord.wav'
    wavfile.write(test_file, sample_rate, (audio * 32767).astype(np.int16))
    print(f"Generated test audio: {test_file}")
    
    return test_file

def main():
    """Main function - analyze audio file or generate test audio."""
    
    # Check if there's an audio file in the current directory
    audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    audio_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in audio_extensions):
            audio_files.append(file)
    
    if audio_files:
        print("Found audio files:")
        for i, file in enumerate(audio_files):
            print(f"{i+1}. {file}")
        
        # Use the first audio file found
        audio_file = audio_files[0]
        print(f"\nAnalyzing: {audio_file}")
        
    else:
        print("No audio files found. Generating test audio...")
        audio_file = generate_test_audio()
        print("Analyzing generated test audio...")
    
    # Perform the analysis
    result = analyze_audio_fourier(audio_file)
    
    if result:
        print(f"\n🎵 Fourier analysis complete!")
        print(f"This shows you what frequencies make up your audio over time.")
    
    return result

if __name__ == "__main__":
    main()