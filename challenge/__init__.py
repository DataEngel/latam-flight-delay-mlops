from importlib import import_module
from typing import Optional

_app: Optional[object] = None


def get_app() -> object:
    """
    La aplicación FastAPI se carga de forma diferida únicamente cuando es solicitada,
    lo que permite posponer dependencias externas hasta el momento de uso.
    """
    global _app
    if _app is None:
        module = import_module("challenge.api.api")
        _app = module.app
    return _app


try:
    app = get_app()
except ImportError:  # pragma: no cover - en entornos sin dependencias opcionales
    app = None

application = app
