# app package
import importlib.util
from pathlib import Path

# Expose the FastAPI app instance from app.py
try:
    _app_py = Path(__file__).parent.parent / "app.py"
    if _app_py.exists():
        _spec = importlib.util.spec_from_file_location("_root_app_module", _app_py)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            if hasattr(_mod, "app"):
                app = getattr(_mod, "app")
except Exception:
    pass
