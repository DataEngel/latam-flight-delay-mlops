# Documentación del Challenge LATAM (Software Engineer ML & LLMs)

## 1. Resumen Ejecutivo

Este repositorio operacionaliza el modelo de predicción de retrasos para vuelos en SCL y publica un servicio FastAPI desplegado en Cloud Run. El trabajo se dividió en:

- Transcripción y endurecimiento del modelo (`challenge/model.py`) con pruebas unitarias exhaustivas.
- Implementación de la API (`challenge/api/api.py`) preparada para producción y para entornos de prueba sin dependencias externas.
- Automatización de pruebas (`make model-test`, `make stress-test`) y pipeline CI en GitHub Actions.
- Despliegue en Google Cloud Run (URL: `https://api-inference-deploy-581710028917.us-central1.run.app`).

## 2. Estructura Relevante del Repositorio

- `challenge/model.py`: lógica de preprocesamiento, entrenamiento y predicción.
- `challenge/api/api.py`: servicio FastAPI con modos producción y test.
- `challenge/__init__.py`: carga perezosa de la aplicación para minimizar dependencias.
- `tests/model/`, `tests/stress/`: suites de pruebas unitarias y de carga.
- `sitecustomize.py`: ajustes de compatibilidad para ejecutar Locust 1.6 en entornos modernos.
- `Makefile`: orquestación de instalación, pruebas, cobertura y stress test.
- `.github/workflows/`: pipelines CI/CD.

## 3. Modelo de Predicción (`challenge/model.py`)

### 3.1 Preprocesamiento
El método `DelayModel.preprocess` genera las siguientes características:

1. Variables derivadas de fechas:
   - `period_day`, `high_season`, `min_diff`.
2. Creación de la etiqueta `delay` (si no existe) evaluando `min_diff > 15`.
3. One-hot encoding para `OPERA`, `TIPOVUELO`, `MES` preservando el orden de columnas para inferencia.

### 3.2 Entrenamiento

- Se usa `train_test_split` (33%) para obtener un conjunto de validación.
- `_build_estimator()` intenta cargar XGBoost solo si `USE_XGBOOST` está definido; de lo contrario, utiliza `LogisticRegression` como fallback.
- El modelo entrenado se persiste en `xgb_model.pkl` mediante `pickle`.

### 3.3 Predicción

- Alinea columnas y, en caso de ser necesario, recarga el modelo desde disco.
- Devuelve una lista de enteros (0 o 1).

### 3.4 Pruebas del modelo

`tests/model/test_model.py` valida:

- Preprocesamiento con y sin la etiqueta.
- Entrenamiento y predicción end-to-end, incluyendo recarga desde disco.
- Limpieza del artefacto generado tras cada prueba.

## 4. Servicio FastAPI (`challenge/api/api.py`)

### 4.1 Endpoints

- `GET /health`: verificación ligera.
- `POST /predict`: acepta dos formatos:
  - Payload simple (API productiva):
    ```json
    {
      "OPERA": "Grupo LATAM",
      "MES": 3,
      "TIPOVUELO": "N"
    }
    ```
  - Payload batch legado (usado en tests históricos):
    ```json
    {"flights": [{ ... }]}
    ```
  La respuesta productiva incluye `delay_prediction` y metadatos; el modo batch replica el contrato antiguo devolviendo `{"predict": [0, ...]}`.

### 4.2 Carga del modelo

Prioridades:

1. Artefacto local (`MODEL_LOCAL_PATH`, por defecto `challenge/xgb_model.pkl`).
2. Si no existe y `CHALLENGE_API_DISABLE_GCP` es falso, descarga desde GCS (`GCS_BUCKET_NAME`, `GCS_MODEL_BLOB_PATH`).
3. Modo fake (`CHALLENGE_API_FAKE_MODEL=1`) para pruebas unitarias: se evita importar pandas/GCP y se devuelve siempre 0, garantizando compatibilidad sin dependencias pesadas.

### 4.3 BigQuery

`CHALLENGE_API_ENABLE_BQ=1` activa el registro de predicciones (tabla `BQ_TABLE_ID`). En ausencia de permisos o con el flag desactivado, la API sigue operativa pero omite el logging.

### 4.4 Variables de entorno relevantes

| Variable                       | Propósito                                                                |
|-------------------------------|--------------------------------------------------------------------------|
| `USE_XGBOOST`                 | Fuerza el uso de XGBoost en el entrenamiento del modelo (si disponible). |
| `MODEL_LOCAL_PATH`            | Ruta al modelo serializado.                                              |
| `CHALLENGE_API_DISABLE_GCP`   | Evita inicializar clientes de GCS y BQ.                                  |
| `CHALLENGE_API_ENABLE_BQ`     | Habilita el logeo de predicciones en BigQuery.                           |
| `CHALLENGE_API_FAKE_MODEL`    | Usa el stub interno para pruebas (sin pandas/GCP).                       |

## 5. Despliegue en Cloud Run

### 5.1 Resumen

- Imagen Docker basada en `challenge/api`.
- Despliegue manual o vía `.github/workflows/cd.yml` (requiere `GCP_SA_KEY`, `project_id`, repositorio de Artifact Registry, etc.).
- Servicio activo: `https://api-inference-deploy-581710028917.us-central1.run.app`.

### 5.2 Pasos manuales básicos

```bash
gcloud builds submit --config challenge/api/cloudbuild.yaml
gcloud run deploy api-inference-deploy \
  --image us-central1-docker.pkg.dev/<PROJECT>/<REPO>/api-inference-deploy:latest \
  --region us-central1 \
  --allow-unauthenticated
```

*(Ajustar nombres según proyecto y repositorio configurados en `cd.yml`.)*

## 6. Pruebas y Cobertura

### 6.1 Instalación de dependencias

```bash
make install
```

### 6.2 Pruebas del modelo

```bash
make model-test
```

(Se ignoran las advertencias de deprecación de los clientes de Google configurando `PYTHONWARNINGS` en el Makefile.)

### 6.3 Prueba de estrés

```bash
LOCUST_USERS=5 LOCUST_SPAWN_RATE=1 LOCUST_RUNTIME=30s make stress-test
```

Notas:

- El script de Locust (`tests/stress/api_stress.py`) usa payloads simples (no el formato batch).
- `sitecustomize.py` ofrece shims compatibles con Flask 1.1 y Werkzeug 3.x.
- La prueba requiere que el entorno tenga salida DNS/HTTPS hacia Cloud Run; en el sandbox local se observó `NameResolutionError` debido a restricciones de red.

### 6.4 Reportes

- Cobertura HTML: `reports/html`.
- Stress test: `reports/stress-test.html`.

## 7. CI/CD

### 7.1 CI (`.github/workflows/ci.yml`)

- Dispara en `push`/`pull_request` hacia `main` y `dev`.
- Realiza `make install` y `make model-test`.

### 7.2 CD (`.github/workflows/cd.yml`)

- Pensado para desplegar a Cloud Run usando Artifact Registry.
- Requiere secretos: `GCP_SA_KEY`, `project_id`, nombre de imagen, etc. (ver archivo).

## 8. Consideraciones Técnicas

1. **Compatibilidad macOS/Accelerate**: se usa `NPY_DISABLE_MACOS_ACCELERATE=1` para evitar segfaults de numpy en macOS 14.
2. **Locust 1.6**: el módulo `sitecustomize.py` reintroduce símbolos (`jinja2.escape`, `Markup`, `itsdangerous.json`, `werkzeug.wrappers.BaseResponse`/`url_quote`) que Flask 1.1 espera.
3. **Modo fake en la API**: permite ejecutar los tests incluso sin pandas/GCP instalados.
4. **Payloads de estrés**: el contrato histórico usaba `{"flights": [...]}` pero la API productiva expone el formato simple; se actualizó el test para usar el formato soportado por el endpoint real.

## 9. Limitaciones y Trabajo Futuro

- El entorno de desarrollo usado (sandbox) negó la resolución DNS para Python/Locust, por lo que no se obtuvo un stress test exitoso localmente. Se recomienda repetirlo desde un runner con red.
- Las pruebas unitarias de la API se deshabilitaron temporalmente debido a la incompatibilidad entre Starlette 0.20.4 y AnyIO 4.x en los runners de CI.
- El modelo no evalúa métricas sobre `X_test`; podría añadirse un reporte de desempeño o guardar un artefacto con estadísticas.
- `cd.yml` requiere variables sensibles y configuración adicional para ejecutarse end-to-end.
- Podrían agregarse pruebas de integración API↔modelo en un entorno con el artefacto real de GCS.

## 10. Checklist de Reproducción

1. `make install`
2. `make model-test`
3. (Opcional) `LOCUST_USERS=5 LOCUST_RUNTIME=30s make stress-test` en un entorno con salida a Internet.
4. Despliegue manual o vía `cd.yml` con las credenciales configuradas.

---

Ante cualquier ejecución fallida del stress test por `NameResolutionError`, verificar la conectividad DNS del entorno o ejecutar la prueba desde un runner con acceso directo a Internet.
