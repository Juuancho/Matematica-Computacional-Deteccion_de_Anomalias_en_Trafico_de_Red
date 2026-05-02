# Detección de anomalías en tráfico de red mediante modelos de espacio de estados y análisis espectral

**Parcial II — Matemáticas Computacionales**

Programa de Ingeniería de Sistemas

Corporación Universitaria Rafael Núñez

---

## Descripción

Este proyecto implementa un sistema de detección de anomalías sobre tráfico de red multivariado (dataset UNSW-NB15), combinando dos enfoques complementarios:

- Una **rama temporal** basada en un *Selective State Space Model* (SSM) con parámetros dependientes de la entrada, que captura dependencias de largo plazo con complejidad lineal en la longitud de la secuencia.
- Una **rama espectral** que extrae componentes periódicos mediante FFT, retiene las top-K frecuencias dominantes y aplica un filtro complejo aprendible por canal antes de regresar al dominio temporal.

Ambas representaciones se combinan mediante una **fusión ponderada** con coeficientes aprendibles, y la salida pasa por una cabeza de clasificación binaria.

Como post-procesamiento opcional, se incluye un **filtro de Kalman escalar** que suaviza la secuencia de probabilidades predichas asumiendo un modelo de random walk sobre el estado latente de anomalía.

---

## Estructura del repositorio

```
.
├── data.py                  # Carga y preprocesamiento del dataset UNSW-NB15
├── model.py                 # Arquitectura: SelectiveSSM, TemporalBlock, SpectralBlock, Fusion, AnomalyDetector
├── train.py                 # Loop de entrenamiento con Adam + CrossEntropyLoss + grad clipping
├── kalman.py                # Filtro de Kalman escalar (KalmanSmoother)
├── notebook.ipynb           # Ejecución completa: datos → entrenamiento → evaluación → Kalman
├── checkpoint.pt            # Pesos del modelo entrenado (mejor F1 de validación)
├── resultados_bloque3.csv   # Imagen que contiene resultados de los 4 filtros Kalman
├── vista_filtros_kalman.png # Visualización en conjunto de los 4 filtros Kalman
├── dataset/
│   └── UNSW_NB15_training-set.csv
└── README.md
```

---

## Dataset

Se utiliza el archivo `UNSW_NB15_training-set.csv` del dataset UNSW-NB15 (Universidad de Nueva Gales del Sur), disponible públicamente en:

- Página oficial: <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
- Kaggle (más fácil de descargar): <https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15>

Tras descargar el CSV, colócalo en la carpeta `dataset/` en la raíz del proyecto. La estructura final debe ser `dataset/UNSW_NB15_training-set.csv`.

---

## Requisitos

- Python ≥ 3.10
- PyTorch ≥ 2.0 (con CUDA si se desea entrenar en GPU)
- NumPy, Pandas, scikit-learn, imbalanced-learn, Matplotlib

### Instalación de dependencias

Se recomienda usar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows

pip install torch numpy pandas scikit-learn imbalanced-learn matplotlib jupyter
```

Para instalar PyTorch con soporte de GPU específico (CUDA 12.x), consulta <https://pytorch.org/get-started/locally/>.

---

## Cómo ejecutar el proyecto

### Opción 1 — Notebook (recomendada)

El notebook `notebook.ipynb` contiene la ejecución completa paso a paso, con todas las celdas ejecutadas y sus salidas visibles. Para abrirlo:

```bash
jupyter notebook notebook.ipynb
```

El flujo del notebook es:

1. Configuración inicial e importaciones.
2. Carga y preprocesamiento del dataset (split 70/15/15, escalado MinMax, ventaneo deslizante, balanceo a nivel de ventana).
3. Instanciación del modelo `AnomalyDetector`.
4. Entrenamiento por 15 épocas (≈ pocos minutos en GPU, más lento en CPU).
5. Demostración de uso programático con `AnomalyDetector.load()`.
6. Evaluación sobre test: baseline vs. filtro de Kalman con varias configuraciones.
7. Tabla comparativa final de las cinco métricas.

### Opción 2 — Solo evaluación con el checkpoint pre-entrenado

Si solo se desea cargar el modelo entrenado y clasificar una ventana nueva:

```python
import torch
from model import AnomalyDetector

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AnomalyDetector.load('checkpoint.pt', map_location=device)
model.to(device)

# nueva_ventana: tensor de shape (1, window_size, input_dim)
prob_anomaly = model.predict_proba(nueva_ventana)
print(f"Probabilidad de anomalía: {prob_anomaly.item():.4f}")
```

### Opción 3 — Entrenar desde cero por línea de comandos

Si se desea re-entrenar el modelo sobrescribiendo el checkpoint:

```python
from data import load_and_preprocess_data, get_dataloaders
from model import AnomalyDetector
from train import train_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds, val_ds, test_ds, input_dim = load_and_preprocess_data(
    'dataset/UNSW_NB15_training-set.csv', window_size=10
)
train_loader, val_loader, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=32)

model = AnomalyDetector(input_dim=input_dim, d_model=64, d_state=16, top_k=5)
train_model(model, train_loader, val_loader, epochs=15, device=device)
```

---

## Hiperparámetros principales

| Parámetro | Valor | Descripción |
|---|---|---|
| `window_size` | 10 | Longitud de cada ventana temporal |
| `d_model` | 64 | Dimensión interna del modelo |
| `d_state` | 16 | Dimensión del estado latente del SSM |
| `top_k` | 5 | Número de componentes frecuenciales retenidas |
| `batch_size` | 32 | Tamaño de batch durante entrenamiento |
| `lr` | 1e-3 | Learning rate del optimizador Adam |
| `epochs` | 15 | Número de épocas de entrenamiento |
| `clip_grad_norm` | 1.0 | Norma máxima para recorte de gradientes |

---

## Resultados

Las métricas obtenidas sobre el conjunto de test (ver notebook para la tabla completa):

| Variante | Accuracy | Recall (macro) | F1 (macro) | MAE | MSE |
|---|---|---|---|---|---|
| Baseline (sin Kalman) | 0.9531 | 0.9535 | 0.9526 | 0.0632 | 0.0357 |
| Kalman — Balanceado | 0.9514 | 0.9515 | 0.9508 | 0.2289 | 0.0797 |
| Kalman — Seguimiento rápido | 0.9531 | 0.9535 | 0.9526 | 0.0637 | 0.0356 |

La discusión detallada de los resultados, el estudio de sensibilidad sobre `(σ²_w, σ²_v)` y las limitaciones del modelo se encuentran en el documento técnico adjunto.

---

## Restricciones de implementación

Conforme al enunciado del parcial, **no se utilizó** ninguna implementación existente de SSM (`mamba-ssm`, `s4-pytorch`, `state-spaces`). Toda la arquitectura fue implementada desde cero a partir de la formulación matemática derivada en el bloque 1 del documento técnico.

---

## Autores

Juan Pablo Hoyos Manjarez

Mauricio José Lugo Granados

Juan Jose Caraballo Nieves

Gabriel Enrique Zambrano Mosquera

Samuel Olivera

Equipo de trabajo del Parcial II — Matemáticas Computacionales, abril de 2026.
