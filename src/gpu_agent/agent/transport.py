"""Bounded, allowlisted transport diagnostics: never serialize exception messages."""

import errno
import socket
import ssl
from typing import Literal

TransportError = Literal[
    "DNS_ERROR",
    "TLS_ERROR",
    "PROXY_ERROR",
    "CONNECTION_REFUSED",
    "CONNECTION_RESET",
    "NETWORK_UNREACHABLE",
    "CONNECT_TIMEOUT",
    "READ_TIMEOUT",
    "WRITE_TIMEOUT",
    "POOL_TIMEOUT",
    "TRANSPORT_TIMEOUT",
    "PROTOCOL_ERROR",
    "CONNECT_ERROR",
    "READ_ERROR",
    "WRITE_ERROR",
    "UNKNOWN_CONNECTION_ERROR",
]


def classify_transport_error(error: BaseException) -> TransportError:
    """Prefer specific nested causes; a category does not establish remote receipt."""
    result: TransportError = "UNKNOWN_CONNECTION_ERROR"
    seen: set[int] = set()
    current: BaseException | None = error
    allowed: dict[str, TransportError] = {
        "ProxyError": "PROXY_ERROR",
        "ConnectTimeout": "CONNECT_TIMEOUT",
        "ReadTimeout": "READ_TIMEOUT",
        "WriteTimeout": "WRITE_TIMEOUT",
        "PoolTimeout": "POOL_TIMEOUT",
        "ConnectError": "CONNECT_ERROR",
        "ReadError": "READ_ERROR",
        "WriteError": "WRITE_ERROR",
        "RemoteProtocolError": "PROTOCOL_ERROR",
        "LocalProtocolError": "PROTOCOL_ERROR",
    }
    for _ in range(12):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if isinstance(current, socket.gaierror):
            return "DNS_ERROR"
        if isinstance(current, ssl.SSLError):
            return "TLS_ERROR"
        if isinstance(current, OSError):
            codes: dict[int, TransportError] = {
                errno.ECONNREFUSED: "CONNECTION_REFUSED",
                errno.ECONNRESET: "CONNECTION_RESET",
                errno.EPIPE: "CONNECTION_RESET",
                errno.ENETUNREACH: "NETWORK_UNREACHABLE",
                errno.EHOSTUNREACH: "NETWORK_UNREACHABLE",
            }
            if current.errno in codes:
                return codes[current.errno]
        if isinstance(current, TimeoutError) and result == "UNKNOWN_CONNECTION_ERROR":
            result = "TRANSPORT_TIMEOUT"
        cls = type(current)
        if cls.__module__.split(".")[0] in {"httpx", "httpx2", "httpcore", "httpcore2"}:
            result = allowed.get(cls.__name__, result)
        current = current.__cause__ or (
            None if current.__suppress_context__ else current.__context__
        )
    return result
