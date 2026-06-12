import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_app_file_exists():
    app_path = Path(__file__).parent.parent / "dashboard" / "app.py"
    assert app_path.exists(), f"dashboard/app.py not found at {app_path}"


def test_streamlit_import():
    import streamlit
    assert streamlit is not None


def test_plotly_import():
    import plotly
    import plotly.graph_objects
    import plotly.express
    assert plotly is not None
