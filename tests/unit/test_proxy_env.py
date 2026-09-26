"""Прокси-переменные, которые не понимает httpx (socks://…), приводятся к рабочему виду или убираются."""

import os

import pytest

from core import proxy_env

NAMES = proxy_env.PROXY_VARIABLES


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)


def test_socks_scheme_is_dropped_without_socksio_and_http_proxy_stays(monkeypatch):
    monkeypatch.setattr(proxy_env.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:2080")
    monkeypatch.setenv("all_proxy", "socks://127.0.0.1:2080")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10809")
    changes = proxy_env.make_httpx_safe()
    assert len(changes) == 2 and all("убрана" in c for c in changes)
    assert "ALL_PROXY" not in os.environ and "all_proxy" not in os.environ
    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:10809"


def test_socks_scheme_becomes_socks5h_with_socksio(monkeypatch):
    monkeypatch.setattr(proxy_env.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:2080")
    proxy_env.make_httpx_safe()
    assert os.environ["ALL_PROXY"] == "socks5h://127.0.0.1:2080"


def test_socks5_without_socksio_is_dropped_too(monkeypatch):
    monkeypatch.setattr(proxy_env.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:2080")
    proxy_env.make_httpx_safe()
    assert "ALL_PROXY" not in os.environ


def test_supported_and_empty_values_are_untouched(monkeypatch):
    monkeypatch.setattr(proxy_env.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "https://proxy.example:3128")
    monkeypatch.setenv("FTP_PROXY", "")
    assert proxy_env.make_httpx_safe() == []
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:1"


def test_httpx_client_can_be_created_after_the_fix(monkeypatch):
    """Ровно то, что падало при загрузке Whisper: httpx.Client(...) с socks:// в окружении."""
    httpx = pytest.importorskip("httpx")
    monkeypatch.setenv("ALL_PROXY", "socks://127.0.0.1:2080")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10809")
    with pytest.raises(ValueError, match="Unknown scheme"):
        httpx.Client()
    proxy_env.make_httpx_safe()
    httpx.Client().close()
