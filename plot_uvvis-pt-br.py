#!/usr/bin/env python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def estimate_color(lam):
    if 380 <= lam < 435: return "Amarelo-Verde (Abs: Violeta)", "yellowgreen"
    elif 435 <= lam < 480: return "Amarelo/Laranja (Abs: Azul)", "orange"
    elif 480 <= lam < 490: return "Laranja/Vermelho (Abs: Verde-Azul)", "orangered"
    elif 490 <= lam < 510: return "Vermelho (Abs: Verde)", "red"
    elif 510 <= lam < 560: return "Púrpura/Magenta (Abs: Amarelo-Verde)", "purple"
    elif 560 <= lam < 590: return "Azul (Abs: Laranja)", "blue"
    elif 590 <= lam < 650: return "Azul-Esverdeado (Abs: Vermelho)", "teal"
    else: return "Incolor/UV", "lightgray"

def main():
    # Tenta localizar o arquivo .stk baseado no nome do .out
    if len(sys.argv) < 2:
        print("Uso: python3 plot_uvvis.py <arquivo.out>")
        sys.exit(1)

    out_file = sys.argv[1]
    stk_file = out_file + ".ABS.stk"

    if not os.path.exists(stk_file):
        print(f"Arquivo {stk_file} não encontrado. Rode: orca_mapspc {out_file} abs")
        sys.exit(1)

    # Lendo o arquivo .stk (Coluna 0: Energia cm-1, Coluna 1: fosc)
    data = np.loadtxt(stk_file)
    
    # Conversão cm-1 para nm: lambda = 10.000.000 / cm-1
    energy_cm1 = data[:, 0]
    fosc = data[:, 1]
    wavelengths = 1e7 / energy_cm1

    # Identificar o pico principal
    max_idx = np.argmax(fosc)
    lambda_max = wavelengths[max_idx]
    f_max = fosc[max_idx]
    cor_texto, cor_hex = estimate_color(lambda_max)

    print(f"\n--- Analise de UV-Vis ---")
    print(f"Lambda Max: {lambda_max:.2f} nm")
    print(f"Osc. Strength: {f_max:.4f}")
    print(f"Cor Estimada: {cor_texto}")

    # Plotagem
    x_nm = np.linspace(300, 800, 1000)
    y_abs = np.zeros_like(x_nm)
    sigma = 15.0 # Alargamento manual para o gráfico

    for wl, f in zip(wavelengths, fosc):
        y_abs += f * np.exp(-0.5 * ((x_nm - wl) / sigma) ** 2)

    plt.figure(figsize=(9, 5))
    plt.plot(x_nm, y_abs, color='black', lw=1.5, label='Espectro Simulado')
    plt.fill_between(x_nm, y_abs, color=cor_hex, alpha=0.3)
    plt.vlines(wavelengths, 0, fosc, colors='red', linestyles='solid', alpha=0.5, label='Transições (Sticks)')

    plt.title(f"UV-Vis: {out_file}\nMax: {lambda_max:.1f} nm -> {cor_texto}")
    plt.xlabel("Comprimento de Onda (nm)")
    plt.ylabel("Intensidade Simulada")
    plt.legend()
    plt.grid(alpha=0.3)
    
    output_img = out_file.replace(".out", "_plot.png")
    plt.savefig(output_img, dpi=300)
    print(f"Gráfico salvo em: {output_img}")
    plt.show()

if __name__ == "__main__":
    main()
