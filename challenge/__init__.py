import os
from importlib import import_module
from typing import Optional

_app: Optional[object] = None


def get_app() -> object:
    """
    La aplicación FastAPI se carga de forma diferida únicamente cuando es requerida,
    de modo que se evitan dependencias externas pesadas durante pruebas centradas en challenge.model.
    """
    global _app
    if _app is None:
        module = import_module("challenge.api.api")
        _app = module.app
    return _app


app = None
if os.getenv("LOAD_CHALLENGE_API", "").lower() in {"1", "true", "yes"}:
    try:
        app = get_app()
    except ImportError:  # pragma: no cover - fallback cuando la API no está disponible
        app = None

application = app
