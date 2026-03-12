#!/usr/bin/env python3
import sys
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def estimate_color(lam):
    """Estimates perceived color based on absorbed wavelength."""
    if 380 <= lam < 435: return "Yellow-Green", "yellowgreen"
    elif 435 <= lam < 480: return "Yellow/Orange", "orange"
    elif 480 <= lam < 490: return "Orange/Red", "orangered"
    elif 490 <= lam < 510: return "Red", "red"
    elif 510 <= lam < 560: return "Purple/Magenta", "purple"
    elif 560 <= lam < 590: return "Blue", "blue"
    elif 590 <= lam < 650: return "Teal", "teal"
    else: return "Colorless/UV", "lightgray"

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_uvvis.py <file.out>")
        sys.exit(1)

    out_file = sys.argv[1]
    stk_file = out_file + ".ABS.stk"

    # Auto-run orca_mapspc if the .stk file is missing
    if not os.path.exists(stk_file):
        print(f"File {stk_file} not found. Running orca_mapspc automatically...")
        try:
            # Executes the native ORCA tool to generate the .stk and .dat files
            subprocess.run(["orca_mapspc", out_file, "abs", "-w500"], check=True)
        except subprocess.CalledProcessError:
            print("Error: orca_mapspc execution failed. Check if the .out file is complete.")
            sys.exit(1)
        except FileNotFoundError:
            print("Error: orca_mapspc command not found. Please ensure ORCA is in your PATH.")
            sys.exit(1)

    # Load data from the generated .stk file
    try:
        data = np.loadtxt(stk_file)
    except Exception as e:
        print(f"Error reading {stk_file}: {e}")
        sys.exit(1)
    
    # Convert Energy in cm-1 to Wavelength in nm (lambda = 1e7 / cm-1)
    energy_cm1 = data[:, 0]
    fosc = data[:, 1]
    wavelengths = 1e7 / energy_cm1

    # Identify the main peak (highest oscillator strength)
    max_idx = np.argmax(fosc)
    lambda_max = wavelengths[max_idx]
    f_max = fosc[max_idx]
    color_text, color_hex = estimate_color(lambda_max)

    print(f"\n--- UV-Vis Analysis ---")
    print(f"Target File: {out_file}")
    print(f"Lambda Max: {lambda_max:.2f} nm")
    print(f"Osc. Strength: {f_max:.4f}")
    print(f"Estimated Color: {color_text}")

    # Plotting setup (Extended range to 200 nm to include deep UV peaks)
    x_nm = np.linspace(200, 800, 2000)
    y_abs = np.zeros_like(x_nm)
    sigma = 15.0 # Broadening factor for the Gaussian curve

    for wl, f in zip(wavelengths, fosc):
        y_abs += f * np.exp(-0.5 * ((x_nm - wl) / sigma) ** 2)

    # Initialize the plot
    plt.figure(figsize=(9, 5))
    plt.plot(x_nm, y_abs, color='black', lw=1.5, label='Simulated Spectrum')
    
    # Fill under the curve with the estimated perceived color
    plt.fill_between(x_nm, y_abs, color=color_hex, alpha=0.3)
    
    # Plot the vertical lines for exact transitions
    plt.vlines(wavelengths, 0, fosc, colors='red', linestyles='solid', alpha=0.5, label='Transitions (Sticks)')

    # Labels and formatting
    plt.title(f"UV-Vis Spectrum: {out_file}\nMax Peak: {lambda_max:.1f} nm -> {color_text}")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Simulated Intensity (a.u.)")
    
    # Force the x-axis to show the 200-800 nm range
    plt.xlim(200, 800) 
    plt.ylim(0, max(max(y_abs), max(fosc)) * 1.1) 
    
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save and display
    output_img = out_file.replace(".out", "_plot.png")
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to: {output_img}\n")
    plt.show()

if __name__ == "__main__":
    main()
