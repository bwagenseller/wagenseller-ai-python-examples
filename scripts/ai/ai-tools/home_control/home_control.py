#!/usr/bin/env python3
"""
home_control - turns lights, switches, fans, covers and the thermostat on and off through Home Assistant (CS-24).

Runs as a tool script (see amadeo_utils/ai/llm/tools/script_tools.py): the tool server calls it with the arguments
on stdin and reads one JSON answer from stdout.

What the model controls
-----------------------
An action and the WORDS it heard ("kitchen lights", "the thermostat"); never an entity_id. Every call reads Home
Assistant's states and registries afresh (one WebSocket connection, see amadeo_utils/home_assistant/ws_client.py)
and resolves the words here, against what the rules below allow:

  1. exact name   - the entity's friendly name or registry name;
  2. alias        - an alias set on the entity in Home Assistant;
  3. area         - an area name or alias, optionally with a kind of device ("kitchen lights", "garage", "all the
                    lights"). An area means every device of that kind in it; with no kind, its lights, switches
                    and fans (never the thermostat, a TV or a siren by accident) - or, for an action those cannot
                    do, the devices that can ("open the garage" means its covers);
  4. fuzzy        - the closest name (difflib), only when it is clearly the best one.

An ambiguous target, or one that matches nothing, is a ToolError naming the close candidates, so the model can ask
"the kitchen ceiling light or the kitchen lamp?".

Skipped before matching: entities that are hidden or disabled (a switch shown as a light hides the original
switch, which would otherwise match twice), config/diagnostic entities, the platforms in 'skip_platforms' (by
default 'alexa_media': its do-not-disturb/shuffle/repeat switches are not house devices), and everything in
'exclude', which is never shown, read or touched.

The rules (a Home Assistant token can do anything, so they live here - see the CS-24 notes)
-------------------------------------------------------------------------------------------
* Allowed by default: every domain not named below.
* Blocked unless the entity_id is in 'explicit_allow': lock, alarm_control_panel, and every cover that is not a
  window covering (device_class blind/shade/curtain/shutter/awning). Garage doors arrive from SmartThings as
  class 'door', so the cover rule fails closed.
* Groupings are checked member by member: a scene or group that lists its members (attribute 'entity_id') is
  allowed only if every member is; one whose members cannot be seen (every SmartThings scene) needs listing.
* An area is resolved to its entities and each is checked on its own; blocked ones are skipped and named.
* Blocked outright, even if listed: script and automation (a script can do anything, unlocking a door included).
* Never called: notify, media_player.play_media, or any Alexa command - only the services in SERVICES below exist.
Blocked entities can still be READ ('status'): knowing the garage door is open is the point of asking.

Security flags: 'physical' - it acts on the house. The tool server never gives it to a delegate worker (which reads
untrusted web pages), and in 'direct' mode switches it off once untrusted text has reached the model. No approval:
the rules above are the safety net. Not 'outbound' (the one configured Home Assistant on the LAN), not 'private'
(decided 2026-09-29), not 'internal_commands' (it does not act on this machine).

The token
---------
A long-lived access token of a NON-admin Home Assistant user, read from 'secrets_file' (the raw token on one line).
The file must be mode 600, or 660 when its group is 'secrets_trusted_group' - the same rule as get_grades.

Config (home_control.json in your tool config folder - outside the repo):
    {
        "base_url": "http://192.168.1.2:8123",                # Home Assistant
        "secrets_file": "/path/to/.home-assistant-token",     # required
        "secrets_trusted_group": "",                          # optional: a group allowed to read it
        "explicit_allow": ["cover.garage_door"],              # locks, alarm panels, non-window covers, opaque scenes
        "exclude": ["switch.server_power"],                   # never shown, read or touched
        "skip_platforms": ["alexa_media"],                    # default ["alexa_media"]
        "request_timeout_s": 8                                # each Home Assistant read/command
    }

Usage:
    home_control.py --describe
    echo '{"action": "turn_off", "target": "kitchen lights"}' | home_control.py --config home_control.json
"""
import difflib
import grp
import math
import os
import re
import stat
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main
from amadeo_utils.home_assistant.ws_client import HomeAssistantError, HomeAssistantWS

ACTIONS = ("turn_on", "turn_off", "toggle", "set_level", "set_temperature", "open", "close", "lock", "unlock",
           "status", "list")
WINDOW_COVERINGS = {"blind", "shade", "curtain", "shutter", "awning"}
NEEDS_LISTING = {"lock", "alarm_control_panel"}                   # plus non-window covers and opaque groupings
BLOCKED_OUTRIGHT = {"script", "automation"}
DEFAULT_SKIP_PLATFORMS = ["alexa_media"]
DEFAULT_REQUEST_TIMEOUT_S = 8
MAX_GROUP_DEPTH = 4                                               # a group of groups of groups... is still checked
FUZZY_ACCEPT = 0.80                                               # best fuzzy score needed to act without asking
FUZZY_MARGIN = 0.06                                               # ...and its lead over the runner-up
FUZZY_SUGGEST = 0.50                                              # scores worth listing as "did you mean"
FALLBACK_TEMPERATURE_RANGE = (45, 90)                             # when a thermostat does not report min/max

# (domain, action) -> (service domain, service). Nothing outside this table can ever be called. set_level and
# set_temperature are handled separately (they need a value); 'scene'/'group' use the generic homeassistant.*.
SERVICES = {
    **{(d, a): (d, a) for d in ("light", "switch", "fan", "input_boolean", "siren", "media_player", "humidifier")
       for a in ("turn_on", "turn_off", "toggle")},
    ("climate", "turn_on"): ("climate", "turn_on"),
    ("climate", "turn_off"): ("climate", "turn_off"),
    ("cover", "open"): ("cover", "open_cover"),
    ("cover", "close"): ("cover", "close_cover"),
    ("cover", "toggle"): ("cover", "toggle"),
    ("lock", "lock"): ("lock", "lock"),
    ("lock", "unlock"): ("lock", "unlock"),
    ("scene", "turn_on"): ("scene", "turn_on"),
    **{("group", a): ("homeassistant", a) for a in ("turn_on", "turn_off", "toggle")},
}
LEVEL_DOMAINS = {"light", "fan", "cover", "media_player"}

# Words naming a kind of device, for area targets ("kitchen lights"). An area target with no kind word gets
# AREA_DEFAULT_DOMAINS: "turn off the basement" must not also turn off its thermostat, TV or siren.
KIND_WORDS = {
    "light": "light", "lights": "light", "lamp": "light", "lamps": "light",
    "switch": "switch", "switches": "switch", "plug": "switch", "plugs": "switch", "outlet": "switch",
    "outlets": "switch",
    "fan": "fan", "fans": "fan",
    "door": "cover", "doors": "cover", "garage door": "cover", "garage doors": "cover", "blind": "cover",
    "blinds": "cover", "shade": "cover", "shades": "cover", "curtain": "cover", "curtains": "cover",
    "lock": "lock", "locks": "lock",
    "thermostat": "climate", "thermostats": "climate", "heat": "climate", "ac": "climate",
    "tv": "media_player", "tvs": "media_player", "television": "media_player",
    "sensor": "sensor", "sensors": "sensor",
}
AREA_DEFAULT_DOMAINS = {"light", "switch", "fan"}
# Status of a whole area shows device-like things only: not updates, events, selects and the like.
STATUS_DOMAINS = {"light", "switch", "fan", "cover", "lock", "climate", "binary_sensor", "sensor", "media_player",
                  "siren", "alarm_control_panel", "input_boolean", "humidifier", "valve", "vacuum"}
# Filler around a target: "all of the", "everything in the".
FILLER_WORDS = {"the", "all", "of", "in", "every", "everything", "my", "our", "please"}
# Attributes worth showing in a status answer.
STATUS_ATTRIBUTES = ("device_class", "unit_of_measurement", "current_temperature", "temperature", "target_temp_low",
                     "target_temp_high", "hvac_action", "current_position", "percentage", "is_volume_muted")

DEFINITION = {
    "name": "home_control",
    "description": "Controls the house through Home Assistant: turns lights, switches, plugs and fans on or off, "
                   "dims lights, opens or closes covers (blinds, garage doors), sets the thermostat, and reports the "
                   "state of devices and sensors. Pass the device or room in the user's own words (\"kitchen lights\", "
                   "\"the basement\", \"thermostat\"); never guess an entity id. Use action 'list' to see what can be "
                   "controlled. If the target is ambiguous the error lists candidates: ask the user which one.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(ACTIONS),
                       "description": "turn_on / turn_off / toggle; set_level (brightness, fan speed or cover "
                                      "position, 0-100); set_temperature (thermostat); open / close (covers); lock / "
                                      "unlock; status (current state); list (everything controllable, by room)."},
            "target": {"type": "string",
                       "description": "What the user named: a device, a room, or a room plus a kind of device. "
                                      "Required for every action except list."},
            "level": {"type": "integer", "description": "set_level only: 0-100 (percent)."},
            "temperature": {"type": "number",
                            "description": "set_temperature only: the target, in the thermostat's own unit."},
        },
        "required": ["action"],
    },
    "flags": {"physical": True},
    "timeout_s": 15,            # a SmartThings command goes HA -> cloud -> hub -> device, typically 1-3 s
}


# ------------------------------------------------------------------------------------------ config and secrets

def group_is_trusted(gid: int, trusted_name: Optional[str]) -> bool:
    """True if 'trusted_name' is set and is the group with this gid."""
    if not trusted_name:
        return False
    try:
        return grp.getgrnam(str(trusted_name)).gr_gid == gid
    except KeyError:
        return False


def read_token(config: Dict[str, Any]) -> str:
    """
    The Home Assistant token from 'secrets_file', after checking who else could read it.

    Only its owner may have access - or, if the config names a 'secrets_trusted_group', also that one group, when it
    is the file's group. Access for "other" users is always refused.

    Raises:
        ToolError: if it is not configured, missing, too widely readable, or empty.
    """
    path = os.path.expanduser(str(config.get("secrets_file") or ""))
    if not path:
        raise ToolError("home_control is not configured: 'secrets_file' is missing from its config")
    try:
        info = os.stat(path)
    except OSError:
        raise ToolError("home_control cannot read its token file (missing, or not readable by the tool server's user)")
    if info.st_mode & stat.S_IRWXO:
        raise ToolError("home_control refuses to run: its token file is readable by every user (chmod o-rwx it)")
    if info.st_mode & stat.S_IRWXG and not group_is_trusted(info.st_gid, config.get("secrets_trusted_group")):
        raise ToolError("home_control refuses to run: its token file is readable by its group, and that group is not "
                        "this tool's secrets_trusted_group (chmod 600 it, or set secrets_trusted_group)")
    try:
        with open(path, encoding="utf-8") as fh:
            token = fh.read().strip()
    except OSError:
        raise ToolError("home_control cannot read its token file")
    if not token:
        raise ToolError("home_control's token file is empty")
    return token


def id_set(config: Dict[str, Any], key: str) -> set:
    """A config list of entity_ids as a set (lower-cased, blanks dropped)."""
    values = config.get(key) or []
    if not isinstance(values, list):
        raise ToolError(f"home_control is misconfigured: '{key}' must be a list of entity ids")
    return {str(v).strip().lower() for v in values if str(v).strip()}


# ------------------------------------------------------------------------------------------ the house model

def normalise(text: str) -> str:
    """Lower case, apostrophes dropped ("Brent's" -> "brents"), punctuation to spaces, single spaces."""
    text = str(text or "").lower().replace("'", "").replace("’", "")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


class Entity:
    """
    One Home Assistant entity, joined from its state and its registry entry.

    Attributes:
        entity_id (str), domain (str), state (str), attributes (dict),
        name (str): the friendly name, as Home Assistant shows it,
        names (set[str]): normalised friendly name and registry name - what an exact match compares against,
        aliases (set[str]): normalised aliases,
        area (str | None): the area's display name (the entity's own area, else its device's).
    """

    def __init__(self, state: Dict[str, Any], entry: Optional[Dict[str, Any]], area: Optional[str]):
        self.entity_id = state["entity_id"].lower()
        self.domain = self.entity_id.split(".", 1)[0]
        self.state = str(state.get("state"))
        self.attributes = state.get("attributes") or {}
        entry = entry or {}
        self.name = str(self.attributes.get("friendly_name") or entry.get("name") or self.entity_id)
        self.names = {normalise(n) for n in (self.name, entry.get("name")) if n}
        self.aliases = {normalise(a) for a in entry.get("aliases") or [] if a}
        self.area = area

    def label(self) -> str:
        """How the entity is named to the model: 'Counter Lights (Kitchen)'."""
        return f"{self.name} ({self.area})" if self.area else self.name


class Match(NamedTuple):
    """
    What a target resolved to.

    Attributes:
        entities (list[Entity]): the device, or the devices in the area.
        area (str | None): the area's name ('the house' for "all the lights"), or None for one named device.
        kind (str | None): the domain a kind word named ("lights" -> 'light'), if any.
    """
    entities: List[Entity]
    area: Optional[str]
    kind: Optional[str]


class House:
    """
    Every entity the agent may know about, after the skips described in the module docstring, plus the areas.

    Args:
        states (list): get_states.
        entities (list): config/entity_registry/list.
        devices (list): config/device_registry/list.
        areas (list): config/area_registry/list.
        config (dict): the tool config (exclude, skip_platforms).
    """

    def __init__(self, states, entities, devices, areas, config):
        self.exclude = id_set(config, "exclude")
        skip_platforms = config.get("skip_platforms", DEFAULT_SKIP_PLATFORMS)
        if not isinstance(skip_platforms, list):
            raise ToolError("home_control is misconfigured: 'skip_platforms' must be a list")
        skip_platforms = {str(p).lower() for p in skip_platforms}

        area_names = {a["area_id"]: a.get("name") or a["area_id"] for a in areas}
        # Area lookup by normalised name or alias -> display name.
        self.area_lookup: Dict[str, str] = {}
        for a in areas:
            for word in [a.get("name")] + list(a.get("aliases") or []):
                if word:
                    self.area_lookup[normalise(word)] = a.get("name") or a["area_id"]
        device_area = {d["id"]: d.get("area_id") for d in devices}
        registry = {e["entity_id"].lower(): e for e in entities}

        # All states, unfiltered, so group members can be checked even when they are hidden.
        self.all_states = {s["entity_id"].lower(): s for s in states}
        self.entities: List[Entity] = []
        for state in states:
            entity_id = state["entity_id"].lower()
            entry = registry.get(entity_id) or {}
            if entity_id in self.exclude or entry.get("hidden_by") or entry.get("disabled_by"):
                continue
            if entry.get("entity_category") in ("config", "diagnostic"):
                continue
            if str(entry.get("platform") or "").lower() in skip_platforms:
                continue
            area_id = entry.get("area_id") or device_area.get(entry.get("device_id"))
            self.entities.append(Entity(state, entry, area_names.get(area_id) if area_id else None))

    # ------------------------------------------------------------------------------------- the rules

    def refusal(self, entity_id: str, explicit_allow: set, depth: int = 0) -> Optional[str]:
        """
        Why the agent may not CONTROL this entity, or None if it may. Groupings are checked member by member.

        Args:
            entity_id: The entity (or a group member).
            explicit_allow: the config's opt-ins.
            depth: How deep inside groupings this check is.

        Returns:
            str | None: a short reason, worded for the model and the user.
        """
        entity_id = entity_id.lower()
        domain = entity_id.split(".", 1)[0]
        if entity_id in self.exclude:
            return "it is excluded from agent control"
        if domain in BLOCKED_OUTRIGHT:
            return f"{domain}s are never run by the agent"
        if entity_id in explicit_allow:
            return None
        state = self.all_states.get(entity_id)
        if state is None:
            return "it is not in Home Assistant"
        attributes = state.get("attributes") or {}
        if domain in NEEDS_LISTING:
            return f"{domain.replace('_', ' ')}s need to be listed in the tool config first"
        if domain == "cover" and attributes.get("device_class") not in WINDOW_COVERINGS:
            return "doors and other non-window covers need to be listed in the tool config first"
        members = attributes.get("entity_id")
        if domain in ("scene", "group") or isinstance(members, list):
            if not isinstance(members, list) or not members:
                return "its members are not visible to Home Assistant, so it needs to be listed in the tool config"
            if depth >= MAX_GROUP_DEPTH:
                return "it is nested too deeply to check"
            for member in members:
                if self.refusal(str(member), explicit_allow, depth + 1):
                    return "one of its members is blocked"
        return None

    # ------------------------------------------------------------------------------------- resolution

    def resolve(self, target: str, domains: Optional[set] = None,
                area_domains: set = AREA_DEFAULT_DOMAINS, allow_many: bool = False) -> "Match":
        """
        The entities a target names, in the order: exact name, alias, area, fuzzy.

        Args:
            target: The words the model passed.
            domains: The domains that can do the action; None means any. A name shared by, say, a light and its
                power sensor then means the light. If no device of these domains has the name, any device that
                has it is returned, so the caller can say why it cannot do the action rather than "not found".
            area_domains: What an area with no kind word means (AREA_DEFAULT_DOMAINS when acting on it).
            allow_many: Several devices sharing the name are all returned instead of being an error. For reading
                only: a SmartThings garage door is both a cover and a binary_sensor called "Third Bay Door".

        Raises:
            ToolError: nothing matched, or several devices share the name - naming the candidates.
        """
        wanted = normalise(target)
        if not wanted:
            raise ToolError("target must name a device or a room")
        pools = [[e for e in self.entities if e.domain in domains]] if domains is not None else []
        pools.append(self.entities)

        for pool in pools:
            for field in ("names", "aliases"):
                hits = [e for e in pool if wanted in getattr(e, field)]
                if len(hits) == 1 or (hits and allow_many):
                    return Match(hits, None, None)
                if len(hits) > 1:
                    raise ToolError(f"'{target}' could mean {self._list(hits)}. Ask the user which one.")

        area_hit = self.area_target(wanted, area_domains)
        if area_hit is not None:
            return area_hit

        return Match([self._fuzzy(target, wanted, pools[0])], None, None)

    def area_target(self, wanted: str, default_domains: set) -> Optional["Match"]:
        """
        Reads a target as [filler] [area] [kind] - "all the kitchen lights", "garage", "the lights". A kind with no
        area means the whole house ("all the lights").

        Returns:
            Match | None: every device of that kind in the area - or of 'default_domains' when no kind was named -
            before the rules; None if the words do not parse that way.
        """
        words = [w for w in wanted.split() if w not in FILLER_WORDS]
        text = " ".join(words)
        kind = None
        for word in sorted(KIND_WORDS, key=len, reverse=True):             # "garage doors" before "doors"
            if text == word or text.endswith(" " + word):
                kind = KIND_WORDS[word]
                text = text[: len(text) - len(word)].strip()
                break
        if text:
            area = self.area_lookup.get(text)
            if area is None:
                return None
        elif kind is not None:
            area = None                                                    # "all the lights": the whole house
        else:
            return None
        domains = {kind} if kind else default_domains
        members = [e for e in self.entities if e.domain in domains and (area is None or e.area == area)]
        where = area or "the house"
        if not members:
            what = f"{kind.replace('_', ' ')}s" if kind else "devices of that kind"
            raise ToolError(f"there are no {what} in {where}")
        return Match(members, where, kind)

    def _fuzzy(self, target: str, wanted: str, pool: List[Entity]) -> Entity:
        """The clearly-best near match by name or alias, else a ToolError listing the close ones."""
        scored = []
        for entity in pool:
            best = max(difflib.SequenceMatcher(None, wanted, n).ratio() for n in entity.names | entity.aliases)
            scored.append((best, entity))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored and scored[0][0] >= FUZZY_ACCEPT and (len(scored) == 1 or scored[0][0] - scored[1][0] >= FUZZY_MARGIN):
            return scored[0][1]
        close = [e for score, e in scored[:5] if score >= FUZZY_SUGGEST]
        if close:
            raise ToolError(f"nothing is called '{target}'. Close matches: {self._list(close)}. Ask the user which one.")
        raise ToolError(f"nothing is called '{target}', and no room has that name. Use action 'list' to see what "
                        "can be controlled.")

    @staticmethod
    def _list(entities: List[Entity]) -> str:
        """'A (Kitchen), B (Office) or C' for an error message; the kind is added where two labels are the same."""
        labels = [e.label() for e in entities]
        labels = [f"{label[:-1]}, {e.domain.replace('_', ' ')})" if labels.count(label) > 1 and label.endswith(")")
                  else f"{label} ({e.domain.replace('_', ' ')})" if labels.count(label) > 1 else label
                  for label, e in zip(labels, entities)]
        return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " or " + labels[-1]


# ------------------------------------------------------------------------------------------ the actions

def describe(entity: Entity) -> Dict[str, Any]:
    """An entity's status for the model: name, area, kind, state and the attributes worth saying."""
    out = {"name": entity.name, "area": entity.area, "kind": entity.domain, "state": entity.state}
    if entity.domain == "light" and isinstance(entity.attributes.get("brightness"), (int, float)):
        out["brightness_pct"] = round(entity.attributes["brightness"] * 100 / 255)
    for key in STATUS_ATTRIBUTES:
        if entity.attributes.get(key) is not None:
            out[key] = entity.attributes[key]
    return out


def action_list(house: House, explicit_allow: set) -> ToolAnswer:
    """Everything controllable, by area. Entities the rules block are shown as such, so the model can explain."""
    by_area: Dict[str, List[Dict[str, Any]]] = {}
    count = 0
    for entity in house.entities:
        if not any(key[0] == entity.domain for key in SERVICES):
            continue                                     # sensors and the like: readable, not controllable
        item = {"name": entity.name, "kind": entity.domain, "state": entity.state}
        reason = house.refusal(entity.entity_id, explicit_allow)
        if reason:
            item["blocked"] = reason
        by_area.setdefault(entity.area or "(no room)", []).append(item)
        count += 1
    rooms = {area: sorted(items, key=lambda i: i["name"]) for area, items in sorted(by_area.items())}
    return ToolAnswer({"rooms": rooms}, f"{count} controllable devices in {len(rooms)} rooms")


def action_status(house: House, target: str) -> ToolAnswer:
    """The state of one device, or of every device-like entity in an area."""
    # An area status without a kind word shows every device-like thing there, sensors included.
    entities, area, _ = house.resolve(target, area_domains=STATUS_DOMAINS, allow_many=True)
    entities = [e for e in entities if e.domain not in BLOCKED_OUTRIGHT]
    if not entities:
        raise ToolError(f"'{target}' is not something the agent can report on")
    result = [describe(e) for e in entities]
    if len(result) == 1:
        one = result[0]
        return ToolAnswer({"devices": result}, f"{one['name']}: {one['state']}")
    if area is None:                                     # one name shared by several devices (a door and its sensor)
        states = ", ".join(f"{r['state']} ({r['kind'].replace('_', ' ')})" for r in result)
        return ToolAnswer({"devices": result}, f"{result[0]['name']}: {states}")
    on = sum(1 for r in result if r["state"] in ("on", "open", "unlocked", "heat", "cool"))
    return ToolAnswer({"area": area, "devices": result}, f"status of {len(result)} devices in {area}, {on} on/open")


def plan_calls(house: House, entities: List[Entity], area: Optional[str], action: str, arguments: Dict[str, Any],
               explicit_allow: set) -> Tuple[List[Tuple[str, str, Dict[str, Any], Entity]], List[str]]:
    """
    Works out the service calls for an action, applying the rules entity by entity.

    Returns:
        (calls, skipped): calls as (service domain, service, service data, entity); skipped as 'Name (why)' for the
        summary. For one named device, a refusal or an unsupported action is a ToolError instead.
    """
    calls, skipped = [], []
    single = area is None
    for entity in entities:
        try:
            calls.append(service_for(entity, action, arguments))
        except ToolError:
            if single:
                raise
            continue                                     # e.g. a fan has no set_temperature: not part of the request
        reason = house.refusal(entity.entity_id, explicit_allow)
        if reason:
            calls.pop()
            if single:
                raise ToolError(f"{entity.label()} cannot be controlled by the agent: {reason}.")
            skipped.append(f"{entity.name} ({reason})")
    if not calls and not skipped:
        raise ToolError(f"nothing in {area} can {action.replace('_', ' ')}")
    return calls, skipped


def service_for(entity: Entity, action: str, arguments: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any], Entity]:
    """
    The one service call that performs 'action' on 'entity', with its values validated and clamped.

    Raises:
        ToolError: if the entity cannot do that action, or a value is missing or not a number.
    """
    domain = entity.domain
    if action == "set_level":
        level = number(arguments, "level", 0, 100)
        if domain not in LEVEL_DOMAINS:
            raise ToolError(f"{entity.label()} has no level to set")
        if domain == "light":
            if entity.attributes.get("supported_color_modes") in (None, ["onoff"]):
                raise ToolError(f"{entity.label()} can only be turned on or off, not dimmed")
            if level == 0:
                return "light", "turn_off", {}, entity
            return "light", "turn_on", {"brightness_pct": level}, entity
        if domain == "fan":
            return "fan", "set_percentage", {"percentage": level}, entity
        if domain == "cover":
            if not int(entity.attributes.get("supported_features") or 0) & 4:     # CoverEntityFeature.SET_POSITION
                raise ToolError(f"{entity.label()} can only open or close")
            return "cover", "set_cover_position", {"position": level}, entity
        return "media_player", "volume_set", {"volume_level": level / 100}, entity
    if action == "set_temperature":
        if domain != "climate":
            raise ToolError(f"{entity.label()} is not a thermostat")
        if entity.state in ("off", "unavailable", "unknown"):
            raise ToolError(f"{entity.label()} is {entity.state}; turn it on first (it needs a heat or cool mode)")
        if entity.attributes.get("temperature") is None and entity.attributes.get("target_temp_low") is not None:
            raise ToolError(f"{entity.label()} is in a heat/cool range mode; set it to heat or cool on the thermostat")
        low = entity.attributes.get("min_temp", FALLBACK_TEMPERATURE_RANGE[0])
        high = entity.attributes.get("max_temp", FALLBACK_TEMPERATURE_RANGE[1])
        return "climate", "set_temperature", {"temperature": number(arguments, "temperature", low, high)}, entity
    key = (domain, action)
    if key not in SERVICES:
        raise ToolError(f"{entity.label()} cannot {action.replace('_', ' ')}")
    service_domain, service = SERVICES[key]
    return service_domain, service, {}, entity


def number(arguments: Dict[str, Any], key: str, low: float, high: float) -> float:
    """A numeric argument, required, clamped to [low, high] (whole numbers stay whole)."""
    value = arguments.get(key)
    if value is None:
        raise ToolError(f"{key} is required for this action")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(str(value))
        except ValueError:
            raise ToolError(f"{key} must be a number")
    if not math.isfinite(value):
        raise ToolError(f"{key} must be a number")
    value = max(low, min(high, value))
    return int(value) if float(value).is_integer() else value


PAST_TENSE = {"turn_on": "Turned on", "turn_off": "Turned off", "toggle": "Toggled", "set_level": "Set",
              "set_temperature": "Set", "open": "Opened", "close": "Closed", "lock": "Locked", "unlock": "Unlocked"}


def run_calls(ha: HomeAssistantWS, calls, action: str, area: Optional[str], skipped: List[str],
              arguments: Dict[str, Any]) -> ToolAnswer:
    """
    Runs the planned calls - one per entity, so one failure does not hide the others - and says what happened.

    The summary names what actually changed ("Turned off 3 in Kitchen"), and the result says when a value was
    clamped, so the model never tells the user "set to 120" when the thermostat stopped at 95.
    """
    done, failed = [], []
    for service_domain, service, data, entity in calls:
        try:
            ha.command("call_service", domain=service_domain, service=service, service_data=data,
                       target={"entity_id": [entity.entity_id]})
            done.append(entity)
        except HomeAssistantError as e:
            failed.append(f"{entity.name} ({e})")
    if not done and failed:
        raise ToolError("Home Assistant refused: " + "; ".join(failed))

    verb = PAST_TENSE[action]
    names = [e.name for e in done]
    value, note = "", None
    if action == "set_level":
        level = number(arguments, "level", 0, 100)
        value = f" to {level}%"
        if level != arguments.get("level"):
            note = f"level {arguments.get('level')} is outside 0-100; used {level}"
    elif action == "set_temperature":
        used = calls[0][2]["temperature"]
        value = f" to {used}"
        if used != arguments.get("temperature"):
            note = f"{arguments.get('temperature')} is outside the thermostat's range; set {used}"
    if area is None:
        summary = f"{verb} {names[0]}{value}" if names else "Nothing changed"
    else:
        summary = f"{verb} {len(names)} in {area}{value}"
    if skipped:
        summary += f"; skipped {len(skipped)}"
    if failed:
        summary += f"; {len(failed)} failed"
    result = {"done": names, "skipped": skipped, "failed": failed}
    if area is not None:
        result["area"] = area
    if note:
        result["note"] = note
    return ToolAnswer(result, summary)


ACTION_DOMAINS = {}                                       # action -> the domains that can do it (one-device matches)
for (_domain, _action) in SERVICES:
    ACTION_DOMAINS.setdefault(_action, set()).add(_domain)
ACTION_DOMAINS["set_level"] = set(LEVEL_DOMAINS)
ACTION_DOMAINS["set_temperature"] = {"climate"}


def handle(arguments: Dict[str, Any], config: Dict[str, Any]):
    """The tool: validate, read the house, resolve the target, apply the rules, act, and report."""
    action = str(arguments.get("action") or "").strip().lower()
    if action not in ACTIONS:
        raise ToolError(f"action must be one of: {', '.join(ACTIONS)}")
    target = arguments.get("target")
    if action != "list" and (not isinstance(target, str) or not target.strip()):
        raise ToolError(f"target is required for {action}: the device or room the user named")
    base_url = str(config.get("base_url") or "").strip()
    if not base_url:
        raise ToolError("home_control is not configured: 'base_url' is missing from its config")
    explicit_allow = id_set(config, "explicit_allow")
    try:
        timeout = float(config.get("request_timeout_s", DEFAULT_REQUEST_TIMEOUT_S))
    except (TypeError, ValueError):
        raise ToolError("home_control is misconfigured: request_timeout_s must be a number")
    token = read_token(config)

    try:
        with HomeAssistantWS(base_url, token, timeout=timeout) as ha:
            house = House(ha.command("get_states"), ha.command("config/entity_registry/list"),
                          ha.command("config/device_registry/list"), ha.command("config/area_registry/list"), config)
            if action == "list":
                return action_list(house, explicit_allow)
            if action == "status":
                return action_status(house, target)
            # An area with no kind word means its lights, switches and fans - or, for an action none of those
            # can do ("open the garage", "lock the porch"), the devices that can.
            domains = ACTION_DOMAINS[action]
            entities, area, _ = house.resolve(target, domains, (AREA_DEFAULT_DOMAINS & domains) or domains)
            calls, skipped = plan_calls(house, entities, area, action, arguments, explicit_allow)
            if not calls:
                raise ToolError(f"nothing was changed: {'; '.join(skipped)}")
            return run_calls(ha, calls, action, area, skipped, arguments)
    except HomeAssistantError as e:
        raise ToolError(f"Home Assistant: {e}")


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
