import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("dl", Path(__file__).parents[1] / "scripts" / "download_data.py")
dl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dl)


def test_months_spans_year_boundary():
    assert dl.months("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]
