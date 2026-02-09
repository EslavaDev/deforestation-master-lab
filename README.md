# Análisis de deforestación — Bosque Nacional Jamanxim

Proyecto de la asignatura **Visión Artificial** (Actividad 3) — UNIR.

Analiza la evolución de la deforestación en el Bosque Nacional Jamanxim (Pará, Brasil) entre 2000 y 2019 mediante técnicas de segmentación de imágenes satelitales Landsat.

## Métodos implementados

- **Preprocesamiento:** CLAHE sobre canal L (CIELab)
- **Segmentación:** Otsu global, umbral adaptativo gaussiano, HSV, CIELab
- **Refinamiento:** operadores morfológicos (apertura + cierre)
- **Cuantificación:** conversión de píxeles a km² (escala: 20 km = 51 px)

## Estructura del proyecto

```
├── analisis_deforestacion.py   # Script principal de análisis
├── frames/                     # 20 imágenes satelitales (2000-2019)
├── output/                     # Figuras y tablas generadas
├── pyproject.toml              # Dependencias del proyecto
└── uv.lock                     # Lock de dependencias
```

## Requisitos

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Uso

```bash
# Instalar dependencias
uv sync

# Ejecutar análisis (genera figuras y tablas en output/)
uv run python analisis_deforestacion.py

# Generar PDF del informe
uv run python md_to_pdf.py
```

## Autor

Daniel Alejandro Eslava Herrera
