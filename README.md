# Image Quality Scoring CLI v0.1

Minimal bundle to compute quality scores for images.

## Estructura
```
image_quality_score/
├── get_image_score.py          # CLI de scoring
└── image_quality/              # Módulo
    ├── __init__.py
    ├── image_quality.py
    ├── recommended_config.json
    └── requirements.txt
```

## Instalación rápida
```bash
cd image_quality_score
python3 -m venv .venv
source .venv/bin/activate
pip install -r image_quality/requirements.txt
```

## Uso
```bash
# Puntuar imágenes sueltas
python get_image_score.py img1.jpg img2.png

# Puntuar un directorio (no recursivo por defecto)
python get_image_score.py --directory /ruta/a/imagenes

# Recursivo y extensiones personalizadas
python get_image_score.py -d /ruta/a/imagenes --recursive --extensions .jpg .png

# Exportar a JSON (por defecto guarda en scores.json)
python get_image_score.py img1.jpg
# Cambiar ruta de salida
python get_image_score.py img1.jpg --output otra_ruta.json
# No guardar JSON
python get_image_score.py img1.jpg --no-save
```

Salida por terminal: muestra `quality_score` (0-1) y sub-scores normalizados de nitidez y exposición para cada imagen.
