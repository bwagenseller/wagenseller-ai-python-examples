"""
A small, standard-library-only client for Home Assistant's WebSocket API (CS-24).

Why WebSocket, and why no library
---------------------------------
Home Assistant keeps entity aliases, areas and the hidden/disabled flags in its registries, and those are only
reachable over the WebSocket API (``config/entity_registry/list`` and friends), not REST. A non-admin user's
long-lived token may read them. The same connection also serves ``get_states`` and ``call_service``, so a caller
needs nothing else.

The agent's tool env ('agent-tools') has no WebSocket package, and Home Assistant's protocol needs very little of
WebSocket: text frames of JSON, one request at a time. So this is a client for exactly that (RFC 6455, client
side): no extensions, no compression, plain ``ws://`` or ``wss://``. It imports only the standard library, so any
conda env can use it.

The Home Assistant protocol, in short
-------------------------------------
1. The server sends ``{"type": "auth_required"}``; the client answers ``{"type": "auth", "access_token": ...}``.
2. The server answers ``auth_ok`` or ``auth_invalid``.
3. Each command is ``{"id": n, "type": "...", ...}`` with a fresh, increasing id. Its answer is
   ``{"id": n, "type": "result", "success": bool, "result" | "error": ...}``. Other messages (events) can arrive in
   between; they are skipped, since this client never subscribes to anything.

Usage::

    with HomeAssistantWS("http://192.168.1.2:8123", token) as ha:
        states = ha.command("get_states")
        ha.command("call_service", domain="light", service="turn_off", target={"entity_id": ["light.desk"]})
"""
import base64
import hashlib
import json
import os
import socket
import ssl
import struct
import urllib.parse
from typing import Any, Dict, Optional

# The fixed GUID from RFC 6455 section 1.3: the server proves it speaks WebSocket by hashing our key with it.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Opcodes (RFC 6455 section 5.2).
_OP_CONTINUATION, _OP_TEXT, _OP_BINARY, _OP_CLOSE, _OP_PING, _OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA

# A registry listing for a large install is a few megabytes; anything far bigger is not Home Assistant talking.
MAX_MESSAGE_BYTES = 64 * 1024 * 1024


class HomeAssistantError(Exception):
    """A connection, protocol, login or command failure. The message never contains the token."""


class HomeAssistantWS:
    """
    One authenticated WebSocket connection to Home Assistant.

    Args:
        base_url (str): Home Assistant's address, e.g. 'http://192.168.1.2:8123' (https gives wss).
        token (str): A long-lived access token.
        timeout (float): Seconds allowed for connecting and for each read.
    """

    def __init__(self, base_url: str, token: str, timeout: float = 10.0):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise HomeAssistantError("the Home Assistant URL must be http:// or https:// with a host name")
        self._host = parsed.hostname
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._tls = parsed.scheme == "https"
        self._path = (parsed.path.rstrip("/") or "") + "/api/websocket"
        self._token = token
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buffer = b""
        self._next_id = 1

    # ------------------------------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "HomeAssistantWS":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def connect(self) -> None:
        """
        Opens the socket, upgrades it to WebSocket and logs in.

        Raises:
            HomeAssistantError: if Home Assistant cannot be reached, does not speak WebSocket, or refuses the token.
        """
        try:
            sock = socket.create_connection((self._host, self._port), timeout=self._timeout)
            if self._tls:
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self._host)
        except OSError as e:
            raise HomeAssistantError(f"cannot reach Home Assistant at {self._host}:{self._port} ({type(e).__name__})")
        self._sock = sock
        self._handshake()
        first = self._receive()
        if first.get("type") != "auth_required":
            raise HomeAssistantError(f"unexpected first message from Home Assistant: {first.get('type')!r}")
        self._send({"type": "auth", "access_token": self._token})
        answer = self._receive()
        if answer.get("type") != "auth_ok":
            raise HomeAssistantError("Home Assistant refused the access token")

    def close(self) -> None:
        """Sends a close frame (best effort) and closes the socket."""
        if self._sock is None:
            return
        try:
            self._send_frame(_OP_CLOSE, struct.pack(">H", 1000))
        except (OSError, HomeAssistantError):
            pass
        try:
            self._sock.close()
        finally:
            self._sock = None

    # ------------------------------------------------------------------------------------------ commands

    def command(self, command_type: str, **fields: Any) -> Any:
        """
        Sends one command and waits for its result.

        Args:
            command_type: e.g. 'get_states', 'config/entity_registry/list', 'call_service'.
            **fields: The command's other fields (for call_service: domain, service, service_data, target).

        Returns:
            The 'result' field of the answer.

        Raises:
            HomeAssistantError: if the command fails, carrying Home Assistant's own error message.
        """
        message_id = self._next_id
        self._next_id += 1
        self._send({"id": message_id, "type": command_type, **fields})
        while True:
            answer = self._receive()
            if answer.get("id") == message_id and answer.get("type") == "result":
                break
        if not answer.get("success"):
            error = answer.get("error") or {}
            raise HomeAssistantError(f"{command_type} failed: {error.get('message') or error.get('code') or 'no reason given'}")
        return answer.get("result")

    # ------------------------------------------------------------------------------------------ WebSocket framing

    def _handshake(self) -> None:
        """The HTTP Upgrade request (RFC 6455 section 4.1), checking the server's Sec-WebSocket-Accept."""
        key = base64.b64encode(os.urandom(16)).decode()
        host = self._host if self._port in (80, 443) else f"{self._host}:{self._port}"
        request = (f"GET {self._path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n")
        self._sock.sendall(request.encode())
        while b"\r\n\r\n" not in self._buffer:
            self._fill()
            if len(self._buffer) > 65536:
                raise HomeAssistantError("Home Assistant's WebSocket handshake reply is too long")
        head, self._buffer = self._buffer.split(b"\r\n\r\n", 1)
        lines = head.decode("latin-1").split("\r\n")
        if len(lines[0].split()) < 2 or lines[0].split()[1] != "101":
            raise HomeAssistantError(f"Home Assistant did not accept the WebSocket upgrade ({lines[0][:60]!r})")
        headers = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
        if headers.get("sec-websocket-accept") != expected:
            raise HomeAssistantError("Home Assistant's WebSocket handshake reply did not match the request")

    def _fill(self) -> None:
        """Reads more bytes into the buffer, or fails if the connection is gone."""
        try:
            chunk = self._sock.recv(65536)
        except socket.timeout:
            raise HomeAssistantError("Home Assistant did not answer in time")
        except OSError as e:
            raise HomeAssistantError(f"the connection to Home Assistant failed ({type(e).__name__})")
        if not chunk:
            raise HomeAssistantError("Home Assistant closed the connection")
        self._buffer += chunk

    def _read_exact(self, count: int) -> bytes:
        """Exactly 'count' bytes from the connection."""
        while len(self._buffer) < count:
            self._fill()
        data, self._buffer = self._buffer[:count], self._buffer[count:]
        return data

    def _receive(self) -> Dict[str, Any]:
        """
        The next whole JSON message: frames are joined, pings answered, and a close frame raises.

        Server frames are never masked (RFC 6455 section 5.1), so no unmasking is done.
        """
        payload = b""
        while True:
            first, second = self._read_exact(2)
            opcode, final = first & 0x0F, bool(first & 0x80)
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read_exact(8))[0]
            if len(payload) + length > MAX_MESSAGE_BYTES:
                raise HomeAssistantError("a message from Home Assistant is too large")
            data = self._read_exact(length)
            if opcode == _OP_PING:
                self._send_frame(_OP_PONG, data)
                continue
            if opcode == _OP_PONG:
                continue
            if opcode == _OP_CLOSE:
                raise HomeAssistantError("Home Assistant closed the connection")
            payload += data
            if final:
                break
        try:
            return json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise HomeAssistantError("Home Assistant sent a message that is not JSON")

    def _send(self, message: Dict[str, Any]) -> None:
        """Sends one JSON message as a text frame."""
        self._send_frame(_OP_TEXT, json.dumps(message).encode("utf-8"))

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        """
        One final frame. Client frames must be masked with a random 4-byte key (RFC 6455 section 5.3).
        """
        mask = os.urandom(4)
        header = bytes([0x80 | opcode])
        if len(payload) < 126:
            header += bytes([0x80 | len(payload)])
        elif len(payload) < 65536:
            header += bytes([0x80 | 126]) + struct.pack(">H", len(payload))
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", len(payload))
        masked = bytes(byte ^ mask[i % 4] for i, byte in enumerate(payload))
        try:
            self._sock.sendall(header + mask + masked)
        except OSError as e:
            raise HomeAssistantError(f"the connection to Home Assistant failed ({type(e).__name__})")
