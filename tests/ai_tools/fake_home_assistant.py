"""
A stand-in Home Assistant for the home_control checks (CS-24): a local WebSocket server with a made-up house.

It speaks just enough of Home Assistant's WebSocket API for the tool - auth, get_states, the entity / device /
area registry listings and call_service - and records every call_service it receives, so a check can assert
exactly what the tool did (and, for a refusal, that it did nothing). Every name in the house is invented.

Usage::

    with FakeHomeAssistant(token="test-token") as ha:
        ... run the tool against ha.url ...
        ha.calls        # [{'domain': 'light', 'service': 'turn_off', 'service_data': {}, 'target': {...}}, ...]
"""
import base64
import hashlib
import http.server
import json
import struct
import threading

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# ------------------------------------------------------------------------------------------ the made-up house

AREAS = [
    {"area_id": "kitchen", "name": "Kitchen", "aliases": ["cookhouse"]},
    {"area_id": "den", "name": "Den", "aliases": []},
    {"area_id": "garage", "name": "Garage", "aliases": []},
    {"area_id": "porch", "name": "Porch", "aliases": []},
    {"area_id": "attic", "name": "Attic", "aliases": []},
]

DEVICES = [
    {"id": "dev_kitchen_ceiling", "area_id": "kitchen"},      # the light gets its area through its device
    {"id": "dev_speaker", "area_id": "kitchen"},
]

# (entity_id, state, attributes, registry entry overrides). friendly_name goes into attributes automatically.
_HOUSE = [
    ("light.kitchen_ceiling", "on", {"friendly_name": "Kitchen Ceiling", "supported_color_modes": ["brightness"],
                                     "brightness": 128}, {"device_id": "dev_kitchen_ceiling"}),
    # The original switch behind a "Show as light" conversion: hidden, and named like the light.
    ("switch.kitchen_ceiling_relay", "on", {"friendly_name": "Kitchen Ceiling"},
     {"hidden_by": "integration", "area_id": "kitchen"}),
    ("light.counter_lamp", "off", {"friendly_name": "Counter Lamp", "supported_color_modes": ["onoff"]},
     {"area_id": "kitchen", "aliases": ["lamp by the sink"]}),
    ("switch.coffee_maker", "off", {"friendly_name": "Coffee Maker"}, {"area_id": "kitchen"}),
    ("sensor.kitchen_temperature", "70", {"friendly_name": "Kitchen Temperature", "unit_of_measurement": "°F",
                                          "device_class": "temperature"}, {"area_id": "kitchen"}),
    ("switch.kitchen_firmware_beta", "off", {"friendly_name": "Kitchen Firmware Beta"},
     {"area_id": "kitchen", "entity_category": "config"}),
    ("switch.kitchen_old_plug", "off", {"friendly_name": "Kitchen Old Plug"},
     {"area_id": "kitchen", "disabled_by": "user"}),
    ("switch.kitchen_speaker_do_not_disturb", "off", {"friendly_name": "Kitchen Speaker Do Not Disturb"},
     {"device_id": "dev_speaker", "platform": "alexa_media"}),
    ("fan.den_fan", "off", {"friendly_name": "Den Fan", "percentage": 0}, {"area_id": "den"}),
    ("climate.thermostat", "heat", {"friendly_name": "Thermostat", "min_temp": 50, "max_temp": 90,
                                    "temperature": 68, "current_temperature": 66}, {"area_id": "den"}),
    ("cover.den_blinds", "open", {"friendly_name": "Den Blinds", "device_class": "blind", "supported_features": 15,
                                  "current_position": 100}, {"area_id": "den"}),
    ("light.reading_lamp_den", "off", {"friendly_name": "Reading Lamp", "supported_color_modes": ["brightness"]},
     {"area_id": "den"}),
    ("light.reading_lamp_porch", "off", {"friendly_name": "Reading Lamp", "supported_color_modes": ["brightness"]},
     {"area_id": "porch"}),
    ("light.porch_light", "off", {"friendly_name": "Porch Light", "supported_color_modes": ["onoff"]},
     {"area_id": "porch"}),
    ("lock.front_door", "locked", {"friendly_name": "Front Door"}, {"area_id": "porch"}),
    ("lock.back_door", "locked", {"friendly_name": "Back Door"}, {"area_id": "porch"}),          # explicit_allow
    ("alarm_control_panel.home_alarm", "armed_away", {"friendly_name": "Home Alarm"}, {"area_id": "porch"}),
    ("cover.garage_door", "closed", {"friendly_name": "Garage Door", "device_class": "door", "supported_features": 3},
     {"area_id": "garage"}),
    # SmartThings gives a garage door opener a contact sensor with the same name.
    ("binary_sensor.garage_door", "off", {"friendly_name": "Garage Door", "device_class": "garage_door"},
     {"area_id": "garage"}),
    ("cover.side_garage_door", "closed", {"friendly_name": "Side Garage Door", "device_class": "garage",
                                          "supported_features": 3}, {"area_id": "garage"}),     # explicit_allow
    ("switch.server_power", "on", {"friendly_name": "Server Power"}, {"area_id": "garage"}),     # exclude
    ("switch.failing_plug", "off", {"friendly_name": "Failing Plug"}, {"area_id": "garage"}),    # call fails
    ("binary_sensor.attic_leak", "off", {"friendly_name": "Attic Leak", "device_class": "moisture"},
     {"area_id": "attic"}),
    # A light group whose members include an excluded switch: refused as a whole.
    ("light.porch_group", "off", {"friendly_name": "Porch Group", "supported_color_modes": ["onoff"],
                                  "entity_id": ["light.porch_light", "switch.server_power"]}, {"area_id": None}),
    ("scene.movie_night", "unknown", {"friendly_name": "Movie Night", "location_id": "made-up-location"}, {}),
    ("scene.evening", "unknown", {"friendly_name": "Evening",
                                  "entity_id": ["light.kitchen_ceiling", "switch.coffee_maker"]}, {}),
    ("scene.lock_up", "unknown", {"friendly_name": "Lock Up",
                                  "entity_id": ["light.counter_lamp", "lock.front_door"]}, {}),
    ("script.unlock_everything", "off", {"friendly_name": "Unlock Everything"}, {}),             # explicit_allow!
    ("automation.night_mode", "on", {"friendly_name": "Night Mode"}, {}),
]

STATES = [{"entity_id": eid, "state": st, "attributes": attrs} for eid, st, attrs, _ in _HOUSE]
ENTITIES = [dict({"entity_id": eid, "name": None, "aliases": [], "area_id": None, "device_id": None,
                  "hidden_by": None, "disabled_by": None, "entity_category": None, "platform": "smartthings"},
                 **overrides) for eid, _, _, overrides in _HOUSE]

# What a tool config for this house looks like (base_url and secrets_file are added by the check).
CONFIG = {
    "explicit_allow": ["lock.back_door", "cover.side_garage_door", "script.unlock_everything"],
    "exclude": ["switch.server_power"],
    "request_timeout_s": 5,
}

FAILING_ENTITY = "switch.failing_plug"


# ------------------------------------------------------------------------------------------ the server

class _Handler(http.server.BaseHTTPRequestHandler):
    """One WebSocket connection: the upgrade, then Home Assistant's message loop."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):                       # keep the check's output clean
        pass

    def do_GET(self):
        if self.path != "/api/websocket" or self.headers.get("Upgrade", "").lower() != "websocket":
            self.send_error(404)
            return
        accept = base64.b64encode(hashlib.sha1((self.headers["Sec-WebSocket-Key"] + _WS_GUID).encode()).digest())
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", accept.decode())
        self.end_headers()
        self.wfile.flush()
        fake = self.server.fake
        self._send({"type": "auth_required"})
        auth = self._receive()
        if auth is None or auth.get("access_token") != fake.token:
            self._send({"type": "auth_invalid", "message": "Invalid access token"})
            return
        self._send({"type": "auth_ok"})
        while True:
            message = self._receive()
            if message is None:
                return
            self._send(fake.answer(message))

    def _receive(self):
        """One client frame (always masked), unmasked and parsed; None when the client closes."""
        head = self.rfile.read(2)
        if len(head) < 2:
            return None
        opcode, length = head[0] & 0x0F, head[1] & 0x7F
        if length == 126:
            length = struct.unpack(">H", self.rfile.read(2))[0]
        elif length == 127:
            length = struct.unpack(">Q", self.rfile.read(8))[0]
        if not head[1] & 0x80:
            raise ValueError("client frame was not masked")         # RFC 6455: a client must mask every frame
        mask = self.rfile.read(4)
        data = bytes(b ^ mask[i % 4] for i, b in enumerate(self.rfile.read(length)))
        if opcode == 0x8:
            return None
        return json.loads(data)

    def _send(self, message):
        """One unmasked text frame."""
        payload = json.dumps(message).encode()
        if len(payload) < 126:
            header = bytes([0x81, len(payload)])
        elif len(payload) < 65536:
            header = bytes([0x81, 126]) + struct.pack(">H", len(payload))
        else:
            header = bytes([0x81, 127]) + struct.pack(">Q", len(payload))
        self.wfile.write(header + payload)
        self.wfile.flush()


class FakeHomeAssistant:
    """
    The fake, on a free local port, in a background thread.

    Attributes:
        url (str): Its base URL, e.g. 'http://127.0.0.1:40123'.
        token (str): The one access token it accepts.
        calls (list[dict]): Every call_service it received, in order.
    """

    def __init__(self, token: str = "fake-token"):
        self.token = token
        self.calls = []
        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._server.fake = self
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def answer(self, message):
        """Home Assistant's answer to one command."""
        kind, mid = message.get("type"), message.get("id")
        results = {"get_states": STATES, "config/entity_registry/list": ENTITIES,
                   "config/device_registry/list": DEVICES, "config/area_registry/list": AREAS}
        if kind in results:
            return {"id": mid, "type": "result", "success": True, "result": results[kind]}
        if kind == "call_service":
            self.calls.append({k: message.get(k) for k in ("domain", "service", "service_data", "target")})
            if FAILING_ENTITY in ((message.get("target") or {}).get("entity_id") or []):
                return {"id": mid, "type": "result", "success": False,
                        "error": {"code": "home_assistant_error", "message": "device did not respond"}}
            return {"id": mid, "type": "result", "success": True, "result": {"context": {"id": "x"}}}
        return {"id": mid, "type": "result", "success": False,
                "error": {"code": "unknown_command", "message": "Unknown command."}}
