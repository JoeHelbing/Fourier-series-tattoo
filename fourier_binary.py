#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def binary_to_signal(binary_string):
    """Convert binary string to digital signal."""
    # Remove spaces and convert to list of integers
    binary_digits = binary_string.replace(' ', '')
    signal = [int(bit) for bit in binary_digits]
    return np.array(signal)

def compute_fourier_series(signal, num_harmonics=10):
    """Compute Fourier series coefficients for the signal."""
    N = len(signal)
    
    # Compute FFT using numpy
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N)
    
    # Get the first few harmonics
    coefficients = []
    for n in range(min(num_harmonics, N)):
        coeff = fft_result[n] / N
        coefficients.append(coeff)
    
    return coefficients, frequencies

def generate_fourier_equation(coefficients):
    """Generate mathematical equation from Fourier coefficients."""
    equation_parts = []
    
    # DC component (n=0)
    a0 = np.real(coefficients[0])
    equation_parts.append(f"{a0:.4f}")
    
    # AC components
    for n in range(1, len(coefficients)):
        if n < len(coefficients):
            coeff = coefficients[n]
            an = 2 * np.real(coeff)  # Cosine coefficient
            bn = -2 * np.imag(coeff)  # Sine coefficient
            
            if abs(an) > 1e-10:  # Only include significant terms
                if an > 0 and len(equation_parts) > 1:
                    equation_parts.append(f" + {an:.4f}*cos({n}*2π*t)")
                else:
                    equation_parts.append(f"{an:.4f}*cos({n}*2π*t)")
            
            if abs(bn) > 1e-10:  # Only include significant terms
                if bn > 0:
                    equation_parts.append(f" + {bn:.4f}*sin({n}*2π*t)")
                else:
                    equation_parts.append(f" - {abs(bn):.4f}*sin({n}*2π*t)")
    
    return "f(t) = " + "".join(equation_parts)

def plot_fourier_analysis(binary_string):
    """Create comprehensive plots of the Fourier analysis."""
    # Convert binary to signal
    signal = binary_to_signal(binary_string)
    t = np.arange(len(signal))
    
    # Compute Fourier coefficients (use more harmonics for better reconstruction)
    coefficients, frequencies = compute_fourier_series(signal, num_harmonics=min(30, len(signal)//2))
    
    # Generate equation
    equation = generate_fourier_equation(coefficients)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Fourier Analysis of 'Doo doo' Binary Signal", fontsize=16)
    
    # Plot 1: Original binary signal
    ax1.step(t, signal, 'b-', linewidth=2, where='mid')
    ax1.set_title('Original Binary Signal')
    ax1.set_xlabel('Bit Position')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.1, 1.1)
    
    # Plot 2: Frequency spectrum (magnitude)
    N = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(N)
    magnitude = np.abs(fft_result)
    
    # Only show positive frequencies
    pos_freqs = freqs[:N//2]
    pos_magnitude = magnitude[:N//2]
    
    ax2.stem(pos_freqs, pos_magnitude, basefmt=' ')
    ax2.set_title('Frequency Spectrum (Magnitude)')
    ax2.set_xlabel('Normalized Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Fourier series reconstruction
    t_fine = np.linspace(0, len(signal)-1, 1000)
    reconstructed = np.real(coefficients[0]) * np.ones_like(t_fine)
    
    # Add harmonics for reconstruction
    for n in range(1, len(coefficients)):
        an = 2 * np.real(coefficients[n])
        bn = -2 * np.imag(coefficients[n])
        reconstructed += an * np.cos(2 * np.pi * n * t_fine / len(signal))
        reconstructed += bn * np.sin(2 * np.pi * n * t_fine / len(signal))
    
    ax3.step(t, signal, 'b-', linewidth=2, label='Original', where='mid', alpha=0.7)
    ax3.plot(t_fine, reconstructed, 'r-', linewidth=2, label='Fourier Reconstruction')
    ax3.set_title('Original vs Fourier Series Reconstruction')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient magnitudes
    n_values = range(len(coefficients))
    magnitudes = [abs(c) for c in coefficients]
    
    ax4.stem(n_values, magnitudes, basefmt=' ')
    ax4.set_title('Fourier Coefficients (Magnitude)')
    ax4.set_xlabel('Harmonic Number')
    ax4.set_ylabel('Magnitude')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('fourier_analysis_doo_doo.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'fourier_analysis_doo_doo.png'")
    
    # Show the plot
    plt.show()
    
    return coefficients, equation

def analyze_fourier(binary_string):
    """Analyze the Fourier series of the binary signal."""
    # Convert binary to signal
    signal = binary_to_signal(binary_string)
    
    # Compute Fourier coefficients
    coefficients, frequencies = compute_fourier_series(signal, num_harmonics=10)
    
    # Generate equation
    equation = generate_fourier_equation(coefficients)
    
    # Print analysis
    print("Fourier Series Equation:")
    print("=" * 50)
    print(equation)
    print("\nBinary string:", binary_string)
    print("Signal length:", len(signal), "bits")
    
    # Print coefficient details
    print("\nFourier Coefficients:")
    print("-" * 30)
    for i, coeff in enumerate(coefficients):
        magnitude = abs(coeff)
        phase = np.angle(coeff)
        print(f"n={i}: magnitude={magnitude:.6f}, phase={phase:.4f} rad")
    
    # Frequency analysis
    print("\nFrequency Analysis:")
    print("-" * 20)
    fundamental_freq = 1.0 / len(signal)
    print(f"Fundamental frequency: {fundamental_freq:.6f}")
    for i in range(1, min(len(coefficients), 6)):
        freq = i * fundamental_freq
        magnitude = abs(coefficients[i])
        if magnitude > 1e-6:  # Only show significant harmonics
            print(f"Harmonic {i}: freq={freq:.6f}, magnitude={magnitude:.6f}")
    
    return coefficients, equation

def compare_harmonics(binary_string):
    """Show how different numbers of harmonics affect reconstruction quality."""
    signal = binary_to_signal(binary_string)
    t = np.arange(len(signal))
    t_fine = np.linspace(0, len(signal)-1, 1000)
    
    # Test different numbers of harmonics
    harmonic_counts = [5, 10, 20, len(signal)//2]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Effect of Number of Harmonics on Reconstruction Quality", fontsize=16)
    
    for i, num_harmonics in enumerate(harmonic_counts):
        ax = axes[i//2, i%2]
        
        # Get coefficients for this number of harmonics
        coefficients, _ = compute_fourier_series(signal, num_harmonics)
        
        # Reconstruct signal
        reconstructed = np.real(coefficients[0]) * np.ones_like(t_fine)
        for n in range(1, len(coefficients)):
            an = 2 * np.real(coefficients[n])
            bn = -2 * np.imag(coefficients[n])
            reconstructed += an * np.cos(2 * np.pi * n * t_fine / len(signal))
            reconstructed += bn * np.sin(2 * np.pi * n * t_fine / len(signal))
        
        # Plot
        ax.step(t, signal, 'b-', linewidth=2, label='Original', where='mid', alpha=0.8)
        ax.plot(t_fine, reconstructed, 'r-', linewidth=2, label=f'Reconstruction ({num_harmonics} harmonics)')
        ax.set_title(f'Using {num_harmonics} Harmonics')
        ax.set_xlabel('Position')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, 1.2)
    
    plt.tight_layout()
    plt.savefig('harmonic_comparison.png', dpi=300, bbox_inches='tight')
    print("Harmonic comparison saved as 'harmonic_comparison.png'")
    plt.show()

def main():
    """Main function to analyze the 'Doo doo' binary."""
    binary_string = "01000100 01101111 01101111 00100000 01100100 01101111 01101111"
    
    print("Fourier Transform Analysis of 'Doo doo' Binary")
    print("=" * 50)
    
    # First show the analysis
    coefficients, equation = analyze_fourier(binary_string)
    
    # Then create plots
    print("\nCreating plots...")
    plot_coefficients, plot_equation = plot_fourier_analysis(binary_string)
    
    # Show harmonic comparison
    print("\nCreating harmonic comparison...")
    compare_harmonics(binary_string)
    
    return coefficients, equation

if __name__ == "__main__":
    main()