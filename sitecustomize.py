"""Runtime adjustments for third-party dependencies used in tooling.

Locust 1.x depends on ``jinja2.escape`` which was removed in Jinja2>=3.0.
When the stress tests run, ensure the symbol exists by aliasing
``markupsafe.escape``. The module is imported automatically by Python when it
is present on ``sys.path`` (PEP 369).
"""

try:  # pragma: no cover - defensive import for tooling
    import jinja2
    from markupsafe import escape as _escape, Markup as _Markup

    if not hasattr(jinja2, "escape"):
        jinja2.escape = _escape
    if not hasattr(jinja2, "Markup"):
        jinja2.Markup = _Markup
except Exception:
    pass

try:  # pragma: no cover - optional compatibility for Flask 1.x
    import itsdangerous
    import json as _json

    if not hasattr(itsdangerous, "json"):
        itsdangerous.json = _json
except Exception:
    pass

try:  # pragma: no cover - compatibility for Werkzeug>=3 with Flask 1.x
    import werkzeug
    from werkzeug.wrappers.response import Response as _Response

    wrappers = getattr(werkzeug, "wrappers", None)
    if wrappers is not None and not hasattr(wrappers, "BaseResponse"):
        wrappers.BaseResponse = _Response
    try:
        from urllib.parse import quote as _quote, unquote as _unquote

        from werkzeug import urls as _urls

        if not hasattr(_urls, "url_quote"):
            _urls.url_quote = _quote
        if not hasattr(_urls, "url_unquote"):
            _urls.url_unquote = _unquote
        if not hasattr(wrappers, "json"):
            import types

            json_module = types.ModuleType("werkzeug.wrappers.json")

            class _JSONMixin:  # pragma: no cover - compatibility shim
                def get_json(self, force=False, silent=False, cache=True):
                    from flask import request

                    return request.get_json(force=force, silent=silent, cache=cache)

            json_module.JSONMixin = _JSONMixin
            wrappers.json = json_module  # type: ignore[attr-defined]
            import sys

            sys.modules.setdefault("werkzeug.wrappers.json", json_module)
    except Exception:
        pass
except Exception:
    pass
