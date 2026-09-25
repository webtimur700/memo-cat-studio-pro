import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def isolated_database(monkeypatch, tmp_path):
    """MainWindow в тестах не должен трогать настоящую базу database/memo_cat.db."""
    import ui.main_window as main_window

    monkeypatch.setattr(main_window, "DB_PATH", tmp_path / "test_memo_cat.db")
