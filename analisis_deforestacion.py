"""
Análisis de la deforestación en el Bosque Nacional Jamanxim (2000-2019)
mediante técnicas de segmentación de imágenes satelitales.

Autor: Daniel Alejandro Eslava Herrera
Asignatura: Visión Artificial - UNIR
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk

# --- Configuración ---
FRAMES_DIR = Path("frames")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

YEARS = list(range(2000, 2020))
NUM_FRAMES = 20
SCALE_KM = 20.0
SCALE_PX = 51.0
KM_PER_PX = SCALE_KM / SCALE_PX
AREA_PER_PX_KM2 = KM_PER_PX ** 2  # km² por píxel

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight"})


def load_images():
    """Carga las 20 imágenes y las convierte a BGR (OpenCV)."""
    images = []
    for i in range(NUM_FRAMES):
        path = FRAMES_DIR / f"frame_{i:02d}.png"
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"No se encontró: {path}")
        images.append(img)
    return images


# =====================================================================
# 1. PRE-PROCESAMIENTO: Mejora de contraste e intensidad
# =====================================================================

def preprocess(img_bgr):
    """Aplica CLAHE en el canal L del espacio Lab para mejorar contraste."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# =====================================================================
# 2. SEGMENTACIÓN: Cuatro métodos
# =====================================================================

def segment_otsu(img_bgr):
    """Segmentación por umbral global de Otsu sobre imagen en escala de grises."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    thresh = threshold_otsu(gray)
    binary = (gray > thresh).astype(np.uint8) * 255
    return binary


def segment_adaptive_otsu(img_bgr, block_size=51):
    """Segmentación por umbral adaptativo (media local) sobre escala de grises."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, 2
    )
    return binary


def segment_hsv(img_bgr):
    """Segmentación basada en color HSV: detecta zonas deforestadas.
    El suelo expuesto presenta tonos marrones/claros con H bajo y V alto.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Suelo expuesto/deforestado: tono amarillo-marrón (H:15-45), brillo alto (V>80)
    lower_deforested = np.array([15, 40, 80])
    upper_deforested = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower_deforested, upper_deforested)
    return mask


def segment_lab(img_bgr):
    """Segmentación basada en color Lab: detecta zonas con alta luminosidad
    y componente a* desplazado, típicas de suelo expuesto.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # Suelo expuesto: L alto (>80), a* > 115 (desplazado hacia rojo vs bosque ~110)
    mask = ((l > 80) & (a > 115) & (b > 145)).astype(np.uint8) * 255
    return mask


# =====================================================================
# 3. POST-PROCESAMIENTO: Operadores morfológicos
# =====================================================================

def morphological_cleanup(binary, selem_radius=3):
    """Aplica apertura y cierre morfológico para eliminar ruido."""
    selem = disk(selem_radius)
    cleaned = opening(binary, selem)
    cleaned = closing(cleaned, selem)
    return cleaned


# =====================================================================
# 4. CÁLCULO DE ÁREA DEFORESTADA
# =====================================================================

def calculate_area_km2(binary_mask):
    """Cuenta píxeles blancos (255) y convierte a km²."""
    pixel_count = np.count_nonzero(binary_mask)
    return pixel_count * AREA_PER_PX_KM2


# =====================================================================
# 5. PIPELINE COMPLETO Y GENERACIÓN DE FIGURAS
# =====================================================================

def run_analysis():
    print("Cargando imágenes...")
    images = load_images()

    methods = {
        "Otsu": segment_otsu,
        "Adaptativo": segment_adaptive_otsu,
        "HSV": segment_hsv,
        "Lab": segment_lab,
    }

    # Almacenar resultados: {método: [área_km2 por año]}
    results = {name: [] for name in methods}

    # Almacenar máscaras para figuras
    all_masks = {name: [] for name in methods}

    for i, img in enumerate(images):
        print(f"  Procesando frame {i:02d} ({YEARS[i]})...")
        img_enhanced = preprocess(img)

        for name, func in methods.items():
            mask_raw = func(img_enhanced)
            mask_clean = morphological_cleanup(mask_raw)
            area = calculate_area_km2(mask_clean)
            results[name].append(area)
            all_masks[name].append(mask_clean)

    # --- FIGURA 1: Mosaico de imágenes originales (selección) ---
    selected_idx = [0, 5, 10, 15, 19]
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
    fig.suptitle("Evolución temporal - Imágenes originales", fontsize=13, fontweight="bold")
    for ax, idx in zip(axes, selected_idx):
        ax.imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(f"{YEARS[idx]}", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_originales.png")
    plt.close(fig)
    print("Figura 1 guardada.")

    # --- FIGURA 2: Pre-procesamiento (antes/después CLAHE) ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle("Efecto del preprocesamiento CLAHE", fontsize=13, fontweight="bold")
    for col, idx in enumerate([0, 10, 19]):
        axes[0, col].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"Original {YEARS[idx]}", fontsize=9)
        axes[0, col].axis("off")
        enhanced = preprocess(images[idx])
        axes[1, col].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        axes[1, col].set_title(f"CLAHE {YEARS[idx]}", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_preprocesamiento.png")
    plt.close(fig)
    print("Figura 2 guardada.")

    # --- FIGURA 3: Comparación de 4 métodos en un año representativo (2010) ---
    ref_idx = 10  # año 2010
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle(f"Comparación de métodos de segmentación ({YEARS[ref_idx]})", fontsize=13, fontweight="bold")
    axes[0].imshow(cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")
    for ax, name in zip(axes[1:], methods):
        ax.imshow(all_masks[name][ref_idx], cmap="gray")
        ax.set_title(f"{name}\n{results[name][ref_idx]:.1f} km²", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_comparacion_metodos.png")
    plt.close(fig)
    print("Figura 3 guardada.")

    # --- FIGURA 4: Evolución segmentación Lab (6 años clave) ---
    key_idx = [0, 4, 8, 12, 16, 19]
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
    fig.suptitle("Evolución de la deforestación - Segmentación Lab", fontsize=13, fontweight="bold")
    for col, idx in enumerate(key_idx):
        axes[0, col].imshow(cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f"{YEARS[idx]}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(all_masks["Lab"][idx], cmap="gray")
        axes[1, col].set_title(f"{results['Lab'][idx]:.1f} km²", fontsize=9)
        axes[1, col].axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_evolucion_lab.png")
    plt.close(fig)
    print("Figura 4 guardada.")

    # --- FIGURA 5: Gráfica de evolución temporal (todos los métodos) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = ["o-", "s--", "^-.", "D:"]
    for (name, areas), marker in zip(results.items(), markers):
        ax.plot(YEARS, areas, marker, label=name, markersize=5, linewidth=1.5)
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Área deforestada (km²)", fontsize=11)
    ax.set_title("Evolución temporal de la deforestación por método de segmentación",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(YEARS)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig5_evolucion_temporal.png")
    plt.close(fig)
    print("Figura 5 guardada.")

    # --- FIGURA 6: Superposición de la máscara sobre la imagen original ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle("Superposición de zonas deforestadas detectadas (Lab)", fontsize=13, fontweight="bold")
    for ax, idx in zip(axes, [0, 10, 19]):
        overlay = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB).copy()
        mask_bool = all_masks["Lab"][idx] > 0
        overlay[mask_bool] = [255, 80, 80]  # rojo semitransparente
        ax.imshow(overlay)
        ax.set_title(f"{YEARS[idx]} - {results['Lab'][idx]:.1f} km²", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_superposicion.png")
    plt.close(fig)
    print("Figura 6 guardada.")

    # --- FIGURA 7: Efecto de operadores morfológicos ---
    ref_idx = 10
    img_enh = preprocess(images[ref_idx])
    mask_raw = segment_lab(img_enh)
    mask_opened = opening(mask_raw, disk(3))
    mask_final = closing(mask_opened, disk(3))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.suptitle(f"Efecto de operadores morfológicos ({YEARS[ref_idx]})", fontsize=13, fontweight="bold")
    titles = ["Original", "Segmentación Lab (cruda)", "Tras apertura", "Tras apertura + cierre"]
    imgs_show = [
        cv2.cvtColor(images[ref_idx], cv2.COLOR_BGR2RGB),
        mask_raw, mask_opened, mask_final
    ]
    cmaps = [None, "gray", "gray", "gray"]
    for ax, title, img_s, cmap in zip(axes, titles, imgs_show, cmaps):
        ax.imshow(img_s, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig7_morfologia.png")
    plt.close(fig)
    print("Figura 7 guardada.")

    # --- TABLA DE RESULTADOS (CSV) ---
    csv_path = OUTPUT_DIR / "resultados.csv"
    with open(csv_path, "w") as f:
        header = "Año," + ",".join(methods.keys())
        f.write(header + "\n")
        for i, year in enumerate(YEARS):
            row = f"{year}," + ",".join(f"{results[name][i]:.2f}" for name in methods)
            f.write(row + "\n")
    print(f"Tabla CSV guardada en {csv_path}")

    # --- TABLA MARKDOWN ---
    md_table_path = OUTPUT_DIR / "tabla_resultados.md"
    with open(md_table_path, "w") as f:
        f.write("| Año | Otsu (km²) | Adaptativo (km²) | HSV (km²) | Lab (km²) |\n")
        f.write("|-----|-----------|------------------|-----------|----------|\n")
        for i, year in enumerate(YEARS):
            row = f"| {year} | {results['Otsu'][i]:.2f} | {results['Adaptativo'][i]:.2f} | {results['HSV'][i]:.2f} | {results['Lab'][i]:.2f} |"
            f.write(row + "\n")
    print(f"Tabla Markdown guardada en {md_table_path}")

    # --- FIGURA 8: Cambio incremental anual ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, areas), marker in zip(results.items(), markers):
        deltas = [areas[i] - areas[i-1] for i in range(1, len(areas))]
        ax.plot(YEARS[1:], deltas, marker, label=name, markersize=4, linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Cambio incremental (km²)", fontsize=11)
    ax.set_title("Cambio incremental anual de la deforestación",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(YEARS[1:])
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig8_incremental.png")
    plt.close(fig)
    print("Figura 8 guardada.")

    # --- FIGURA 9: Porcentaje de deforestación respecto al área total ---
    total_area = 720 * 800 * AREA_PER_PX_KM2
    fig, ax = plt.subplots(figsize=(10, 5))
    for (name, areas), marker in zip(results.items(), markers):
        pcts = [a / total_area * 100 for a in areas]
        ax.plot(YEARS, pcts, marker, label=name, markersize=5, linewidth=1.5)
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Deforestación (% del área total)", fontsize=11)
    ax.set_title(f"Porcentaje de deforestación respecto al área total (~{total_area:.0f} km²)",
                  fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(YEARS)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig9_porcentaje.png")
    plt.close(fig)
    print("Figura 9 guardada.")

    # --- FIGURA 10: Barras comparativas para años representativos ---
    rep_years = [2000, 2005, 2010, 2015, 2019]
    rep_idx = [y - 2000 for y in rep_years]
    x = np.arange(len(rep_years))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(methods):
        vals = [results[name][i] for i in rep_idx]
        ax.bar(x + j * width, vals, width, label=name)
    ax.set_xlabel("Año", fontsize=11)
    ax.set_ylabel("Área deforestada (km²)", fontsize=11)
    ax.set_title("Comparación de métodos para años representativos",
                  fontsize=12, fontweight="bold")
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(rep_years)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig10_barras.png")
    plt.close(fig)
    print("Figura 10 guardada.")

    # --- FIGURA 11: Overlay Otsu para años seleccionados ---
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle("Zonas deforestadas detectadas (Otsu) - overlay rojo", fontsize=13, fontweight="bold")
    for ax, idx in zip(axes, [0, 5, 10, 15, 19]):
        overlay = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB).copy()
        mask_bool = all_masks["Otsu"][idx] > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.4).astype(np.uint8)
        overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"{YEARS[idx]} - {results['Otsu'][idx]:.0f} km²", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig11_overlay_otsu.png")
    plt.close(fig)
    print("Figura 11 guardada.")

    # --- FIGURA 12: Overlay HSV para años seleccionados ---
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle("Zonas deforestadas detectadas (HSV) - overlay rojo", fontsize=13, fontweight="bold")
    for ax, idx in zip(axes, [0, 5, 10, 15, 19]):
        overlay = cv2.cvtColor(images[idx], cv2.COLOR_BGR2RGB).copy()
        mask_bool = all_masks["HSV"][idx] > 0
        overlay[mask_bool, 0] = np.clip(overlay[mask_bool, 0].astype(int) + 120, 0, 255).astype(np.uint8)
        overlay[mask_bool, 1] = (overlay[mask_bool, 1] * 0.4).astype(np.uint8)
        overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.4).astype(np.uint8)
        ax.imshow(overlay)
        ax.set_title(f"{YEARS[idx]} - {results['HSV'][idx]:.0f} km²", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig12_overlay_hsv.png")
    plt.close(fig)
    print("Figura 12 guardada.")

    # --- FIGURA 13: Grilla comparativa 4 métodos para año 2000 ---
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    fig.suptitle(f"Comparación de métodos de segmentación ({YEARS[0]})", fontsize=13, fontweight="bold")
    axes[0].imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")
    for ax, name in zip(axes[1:], methods):
        ax.imshow(all_masks[name][0], cmap="gray")
        ax.set_title(f"{name}\n{results[name][0]:.1f} km²", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig13_comp_2000.png")
    plt.close(fig)
    print("Figura 13 guardada.")

    # --- FIGURA 14: Histogramas del canal L antes y después de CLAHE ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Histogramas del canal L (CIELab) - antes y después de CLAHE", fontsize=13, fontweight="bold")
    for ax, idx in zip(axes, [0, 10, 19]):
        lab_orig = cv2.cvtColor(images[idx], cv2.COLOR_BGR2LAB)
        enhanced = preprocess(images[idx])
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        ax.hist(lab_orig[:,:,0].ravel(), bins=128, alpha=0.6, label="Original", color="steelblue", density=True)
        ax.hist(lab_enh[:,:,0].ravel(), bins=128, alpha=0.6, label="CLAHE", color="darkorange", density=True)
        ax.set_title(f"Año {YEARS[idx]}", fontsize=10)
        ax.set_xlabel("Luminancia (L)")
        ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig14_histogramas.png")
    plt.close(fig)
    print("Figura 14 guardada.")

    # --- TABLA COMPLETA CON PIXELES ---
    md_full_path = OUTPUT_DIR / "tabla_completa.md"
    with open(md_full_path, "w") as f:
        f.write("| Año | Otsu (px) | Otsu (km²) | Adapt. (px) | Adapt. (km²) | HSV (px) | HSV (km²) | Lab (px) | Lab (km²) |\n")
        f.write("|-----|----------|-----------|------------|-------------|---------|----------|---------|----------|\n")
        for i, year in enumerate(YEARS):
            row_parts = [f"| {year}"]
            for name in methods:
                area = results[name][i]
                px = int(area / AREA_PER_PX_KM2)
                row_parts.append(f" {px:,} | {area:,.2f}")
            f.write(" |".join(row_parts) + " |\n")
    print(f"Tabla completa guardada en {md_full_path}")

    # --- Resumen final ---
    print(f"\n=== RESUMEN ===")
    print(f"Escala: {SCALE_KM} km = {SCALE_PX} px → {AREA_PER_PX_KM2:.6f} km²/px")
    print(f"Área total imagen: {total_area:.0f} km²")
    for name in methods:
        print(f"\n{name}:")
        print(f"  Año 2000: {results[name][0]:.2f} km² ({results[name][0]/total_area*100:.2f}%)")
        print(f"  Año 2019: {results[name][-1]:.2f} km² ({results[name][-1]/total_area*100:.2f}%)")
        print(f"  Incremento: {results[name][-1] - results[name][0]:.2f} km²")

    return results


if __name__ == "__main__":
    run_analysis()
