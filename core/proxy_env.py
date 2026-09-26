"""Прокси в переменных окружения, которые не понимает httpx.

httpx (им ходят huggingface_hub и faster-whisper) при создании клиента разбирает ALL_PROXY/HTTP(S)_PROXY и падает
`ValueError: Unknown scheme for proxy URL 'socks://…'` на схемах вроде `socks://` (их выдают VPN-клиенты), а `socks5://`
работает только с пакетом socksio. Клиент не создаётся вообще, даже если обычный HTTP(S)_PROXY рабочий и был бы достаточен.

`make_httpx_safe()` меняет окружение ТЕКУЩЕГО процесса: `socks://`/`socks5://` превращаются в `socks5h://`, если socksio есть,
иначе такая переменная убирается (остальные — HTTP(S)_PROXY, no_proxy — не трогаются). Возвращает описание сделанного для лога.
Приложение к LM Studio это не касается: `llm/*` ходит через urllib в обход прокси.
"""

from __future__ import annotations

import importlib.util
import os

PROXY_VARIABLES = ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")   # те, что читает httpx
_HTTPX_SCHEMES = ("http", "https", "socks5", "socks5h")


def make_httpx_safe() -> list[str]:
    have_socksio = importlib.util.find_spec("socksio") is not None
    changes: list[str] = []
    for name in PROXY_VARIABLES:
        value = os.environ.get(name, "").strip()
        scheme, separator, rest = value.partition("://")
        if not value or not separator:
            continue
        scheme = scheme.lower()
        is_socks = scheme.startswith("socks")
        if scheme in ("http", "https") or (scheme in _HTTPX_SCHEMES and (not is_socks or have_socksio)):
            continue
        if is_socks and have_socksio:
            os.environ[name] = f"socks5h://{rest}"
            changes.append(f"{name}: {scheme}:// -> socks5h://")
        else:
            del os.environ[name]
            changes.append(f"{name}: {scheme}:// убрана (httpx её не поддерживает без пакета socksio)")
    return changes
