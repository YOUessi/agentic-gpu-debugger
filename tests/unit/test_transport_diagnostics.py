import errno
import socket
import ssl

import httpx2
import pytest

from gpu_agent.agent.transport import classify_transport_error


@pytest.mark.parametrize(
    "cause,code",
    [
        (socket.gaierror(-2, "secret-canary"), "DNS_ERROR"),
        (ssl.SSLError("secret-canary"), "TLS_ERROR"),
        (OSError(errno.ECONNREFUSED, "secret-canary"), "CONNECTION_REFUSED"),
        (OSError(errno.ECONNRESET, "secret-canary"), "CONNECTION_RESET"),
        (OSError(errno.ENETUNREACH, "secret-canary"), "NETWORK_UNREACHABLE"),
        (httpx2.ProxyError("secret-canary"), "PROXY_ERROR"),
        (httpx2.ConnectTimeout("secret-canary"), "CONNECT_TIMEOUT"),
        (httpx2.ReadTimeout("secret-canary"), "READ_TIMEOUT"),
        (httpx2.WriteTimeout("secret-canary"), "WRITE_TIMEOUT"),
        (httpx2.PoolTimeout("secret-canary"), "POOL_TIMEOUT"),
        (httpx2.RemoteProtocolError("secret-canary"), "PROTOCOL_ERROR"),
        (RuntimeError("secret-canary"), "UNKNOWN_CONNECTION_ERROR"),
    ],
)
def test_safe_categories(cause, code):
    wrapper = RuntimeError("https://user:secret-canary@example.invalid")
    wrapper.__cause__ = cause
    assert classify_transport_error(wrapper) == code


def test_cycles_and_deep_dns_cause():
    outer = httpx2.ConnectError("secret-canary")
    inner = socket.gaierror(-2, "secret-canary")
    outer.__cause__ = inner
    assert classify_transport_error(outer) == "DNS_ERROR"
    outer.__cause__ = outer
    assert classify_transport_error(outer) == "CONNECT_ERROR"
