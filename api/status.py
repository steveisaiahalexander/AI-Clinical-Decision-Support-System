"""Sanitized component readiness checks for the development status endpoint."""

import ipaddress
from typing import Any, Dict
from urllib.parse import urlparse


def _is_loopback(host: str) -> bool:
    if host.casefold() == "localhost" or host.casefold().endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def probe_ollama(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Check the local tags endpoint without generating text or exposing errors."""
    model = str(settings.get("model", ""))
    if settings.get("provider") != "ollama":
        return {"status": "unavailable", "message": "Optional Ollama provider is disabled.", "details": {"configured_model": model or None}}

    parsed = urlparse(str(settings.get("base_url", "")))
    local = bool(parsed.hostname and _is_loopback(parsed.hostname))
    permitted = (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and not parsed.username
        and not parsed.password
        and (local or (settings.get("allow_remote", False) and parsed.scheme == "https"))
    )
    if not permitted:
        return {"status": "unavailable", "message": "Configured Ollama endpoint is not available for status checks.", "details": {"configured_model": model or None}}

    try:
        import httpx

        response = httpx.get(
            f"{str(settings['base_url']).rstrip('/')}/api/tags",
            timeout=2.0,
            trust_env=False,
            follow_redirects=False,
        )
        if response.status_code != 200:
            return {"status": "unavailable", "message": "Ollama did not respond successfully.", "details": {"configured_model": model or None}}
        available = response.json().get("models", [])
        names = {str(item.get("name", "")) for item in available if isinstance(item, dict)}
    except Exception:
        return {"status": "unavailable", "message": "Ollama is not reachable.", "details": {"configured_model": model or None}}

    if model in names:
        return {"status": "ready", "message": "Configured local model is available.", "details": {"configured_model": model}}
    return {"status": "degraded", "message": "Ollama is reachable, but the configured model is not listed.", "details": {"configured_model": model or None}}
