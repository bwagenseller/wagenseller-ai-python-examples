#!/usr/bin/env python3
"""
get_weather - the forecast from the US National Weather Service (CS-21 tool).

Runs in the 'agent-tools' conda env as a tool script (see amadeo_utils/ai/llm/tools/script_tools.py):
the tool server calls it with the arguments on stdin and reads one JSON answer from stdout.

Where the forecast is for, in order of preference:
  1. 'latitude' and 'longitude', if the model gives both;
  2. 'zip', a 5-digit US ZIP code, looked up OFFLINE with pgeocode (no geocoding service is
     contacted - the table is cached locally when the env is set up);
  3. the home location in this tool's config file.

What comes back:
  * detail "daily" (the default) - the NWS 12-hour periods ("Tonight", "Friday", ...);
  * detail "hourly" - one entry per hour, for the next 'hours' hours (at most 48);
  * in both, the forecast precipitation AMOUNTS covering the same window. The NWS forecasts
    amounts only in blocks of (usually) 6 hours, so they are reported in those blocks, as
    published - never spread over single hours, which would invent numbers the forecast does
    not contain. Snow and ice amounts appear only when forecast.

The forecast comes from api.weather.gov, which needs no API key but asks every client to
identify itself with a User-Agent that includes a contact - that goes in the config, not here.
NWS covers the United States and its territories only. The amounts come from a separate
request (the raw gridpoint data); if that one fails, the forecast is still returned, with a
note that amounts are unavailable.

Security flags: 'outbound' - the coordinates are sent to api.weather.gov. The result is public
weather data, written by the NWS, and is not treated as private.

Config (get_weather.json in your tool config folder - outside the repo):
    {
        "user_agent": "(home-assistant, you@example.com)",   # NWS asks for a contact
        "home_latitude": 40.0,                                # optional fallback location
        "home_longitude": -75.0,
        "home_name": "Home"
    }

Usage:
    get_weather.py --describe
    echo '{"zip": "10001"}' | get_weather.py --config get_weather.json
    echo '{"detail": "hourly", "hours": 24}' | get_weather.py --config get_weather.json
"""
import math
import re
from datetime import datetime, timedelta, timezone

import requests

from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main

NWS_BASE = "https://api.weather.gov"
DEFAULT_USER_AGENT = "(amadeo-tool-server get_weather)"
DEFAULT_PERIODS = 4
MAX_PERIODS = 14            # NWS returns 14 twelve-hour periods: seven days and nights
DEFAULT_HOURS = 12
MAX_HOURS = 48              # keeps an hourly answer well inside the per-result token cap
DETAILS = ("daily", "hourly")
MM_PER_INCH = 25.4
HTTP_TIMEOUT = (5, 10)      # (connect, read) seconds, per request
# The gridpoint data (for the rainfall amounts) is by far the largest NWS response and the slowest: it once timed out
# at a 10 s read (2026-09-25). It gets longer, and the tool's own timeout_s leaves room for it: a slow gridpoint must
# cost the amounts at worst, never the whole call - the server kills a tool that overruns timeout_s.
GRID_HTTP_TIMEOUT = (5, 20)

# The gridpoint layers reported as amounts, and the key each gets in a block.
AMOUNT_LAYERS = (
    ("quantitativePrecipitation", "precip_in"),   # liquid equivalent: rain, plus melted snow/ice
    ("snowfallAmount", "snow_in"),
    ("iceAccumulation", "ice_in"),
)

DEFINITION = {
    "name": "get_weather",
    "description": "Gets the National Weather Service forecast for a US location: by latitude and longitude, "
                   "by 5-digit US ZIP code, or - if neither is given - for home. Daily (12-hour periods) or "
                   "hourly; both include forecast precipitation amounts in inches (in 6-hour blocks).",
    "parameters": {
        "type": "object",
        "properties": {
            "latitude": {"type": "number", "description": "Latitude in degrees; give with longitude."},
            "longitude": {"type": "number", "description": "Longitude in degrees; give with latitude."},
            "zip": {"type": "string", "description": "A 5-digit US ZIP code, e.g. 10001."},
            "detail": {"type": "string", "enum": list(DETAILS),
                       "description": "'daily' for 12-hour periods (default) or 'hourly' for hour by hour."},
            "periods": {"type": "integer", "description": f"daily only: how many 12-hour periods (1-{MAX_PERIODS}). Default {DEFAULT_PERIODS}."},
            "hours": {"type": "integer", "description": f"hourly only: how many hours ahead (1-{MAX_HOURS}). Default {DEFAULT_HOURS}."},
        },
        "required": [],
    },
    "flags": {"outbound": True},
    # Must cover the worst case of all three requests - points, forecast (HTTP_TIMEOUT each) and gridpoint
    # (GRID_HTTP_TIMEOUT): 15 + 15 + 25 = 55 s - so a slow gridpoint costs the amounts, never the whole answer.
    "timeout_s": 60,
}


def resolve_location(arguments, config):
    """
    Works out where the forecast is for.

    Returns:
        tuple[float, float, str]: latitude, longitude, and a label for how the place was chosen.

    Raises:
        ToolError: for coordinates out of range, an unknown ZIP, or no location at all.
    """
    lat, lon = arguments.get("latitude"), arguments.get("longitude")
    if (lat is None) != (lon is None):
        raise ToolError("give both latitude and longitude, or neither")
    if lat is not None:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            raise ToolError("latitude and longitude must be numbers")
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ToolError("latitude must be -90..90 and longitude -180..180")
        return lat, lon, f"{lat:.4f}, {lon:.4f}"

    zip_code = arguments.get("zip")
    if zip_code:
        zip_code = str(zip_code).strip()
        if not re.fullmatch(r"\d{5}", zip_code):
            raise ToolError("zip must be a 5-digit US ZIP code")
        import pgeocode       # deferred: it loads pandas, which only this branch needs
        found = pgeocode.Nominatim("us").query_postal_code(zip_code)
        if found is None or any(isinstance(v, float) and math.isnan(v) for v in (found.latitude, found.longitude)):
            raise ToolError(f"unknown US ZIP code {zip_code}")
        return float(found.latitude), float(found.longitude), f"ZIP {zip_code} ({found.place_name}, {found.state_code})"

    if config.get("home_latitude") is not None and config.get("home_longitude") is not None:
        return float(config["home_latitude"]), float(config["home_longitude"]), config.get("home_name", "home")
    raise ToolError("no location given, and no home location is configured")


def bounded_int(arguments, name, default, maximum):
    """
    Reads a whole-number argument and clamps it to 1..maximum.

    Raises:
        ToolError: if the value is not a whole number.
    """
    try:
        value = int(arguments.get(name, default))
    except (TypeError, ValueError):
        raise ToolError(f"{name} must be a whole number")
    return max(1, min(maximum, value))


def nws_get(url, config, timeout=HTTP_TIMEOUT):
    """One GET against api.weather.gov, with the identifying User-Agent it requires. 'timeout' is (connect, read)."""
    headers = {"User-Agent": config.get("user_agent") or DEFAULT_USER_AGENT, "Accept": "application/geo+json"}
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
    except requests.RequestException as e:
        raise ToolError(f"the weather service could not be reached ({type(e).__name__})")
    if response.status_code == 404:
        raise ToolError("no forecast for that location - the National Weather Service covers the US only")
    if response.status_code != 200:
        raise ToolError(f"the weather service answered HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        raise ToolError("the weather service sent something that was not JSON")


def nws_url(properties, key):
    """
    A follow-on URL from the /points answer, accepted only if it points back at api.weather.gov.

    Raises:
        ToolError: if the URL is missing or points anywhere else.
    """
    url = properties.get(key)
    if not url or not url.startswith(NWS_BASE + "/"):
        raise ToolError("the weather service gave no forecast for that location")
    return url


# ------------------------------------------------------------------------------------------ time helpers

_DURATION = re.compile(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?)?$")


def parse_valid_time(valid_time):
    """
    Parses an NWS gridpoint 'validTime' - an ISO 8601 interval "start/duration", e.g.
    "2026-09-26T06:00:00+00:00/PT6H" or ".../P1DT6H".

    Returns:
        tuple[datetime, datetime] | None: aware start and end, or None if it cannot be parsed.
    """
    try:
        start_text, duration_text = valid_time.split("/", 1)
        start = datetime.fromisoformat(start_text)
    except (AttributeError, ValueError):
        return None
    match = _DURATION.match(duration_text)
    if not match or not any(match.groups()) or start.tzinfo is None:
        return None
    days, hours, minutes = (int(g or 0) for g in match.groups())
    end = start + timedelta(days=days, hours=hours, minutes=minutes)
    return (start, end) if end > start else None


def local_zone(time_zone_name, fallback_offset):
    """
    The forecast location's time zone: the IANA zone the NWS names, or - if that is missing or this
    Python has no zone database - the fixed UTC offset seen on the forecast's own timestamps.
    """
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(time_zone_name)
    except Exception:
        return fallback_offset or timezone.utc


def clock(moment, zone, with_day=True):
    """A short local time for people and models: "Fri 3 PM", "Fri 3:30 PM", or just "3 PM"."""
    local = moment.astimezone(zone)
    hour = local.hour % 12 or 12
    minutes = f":{local.minute:02d}" if local.minute else ""
    text = f"{hour}{minutes} {'AM' if local.hour < 12 else 'PM'}"
    return f"{local:%a} {text}" if with_day else text


def span(start, end, zone):
    """ "Fri 2 AM-8 AM" within one day, "Fri 8 PM-Sat 2 AM" across midnight."""
    same_day = start.astimezone(zone).date() == end.astimezone(zone).date()
    return f"{clock(start, zone)}-{clock(end, zone, with_day=not same_day)}"


def inches(mm):
    """Millimetres to inches, to the hundredth."""
    return round(mm / MM_PER_INCH, 2)


def leading_int(text):
    """The number at the start of "73°F" / "30%", or None."""
    match = re.match(r"-?\d+", text or "")
    return int(match.group()) if match else None


# ------------------------------------------------------------------------------------------ amounts

def precipitation_blocks(grid_properties, window_start, window_end, zone):
    """
    The forecast amounts whose blocks overlap [window_start, window_end), in the NWS's own blocks.

    A block that straddles an edge of the window is kept whole - its amount cannot honestly be
    split - so the total covers from the first kept block's start to the last one's end, and says so.

    Args:
        grid_properties (dict): 'properties' of the /gridpoints answer.
        window_start, window_end (datetime): aware; the span the returned forecast covers.
        zone: time zone for the labels.

    Returns:
        dict: {"blocks": [only blocks with some amount], "total": {...amounts, "from", "to"} or None if no
            block overlaps the window, "complete": False if the published amounts stop before the window
            ends}. Values are inches; snow/ice keys appear only when forecast.
    """
    merged = {}                                   # (start, end) -> {key: mm}
    for layer, key in AMOUNT_LAYERS:
        data = grid_properties.get(layer) or {}
        if data.get("uom", "wmoUnit:mm") != "wmoUnit:mm":     # every layer is published in mm; never guess
            continue
        for entry in data.get("values") or []:
            interval = parse_valid_time(entry.get("validTime"))
            amount = entry.get("value")
            if interval is None or not isinstance(amount, (int, float)):
                continue
            if interval[1] <= window_start or interval[0] >= window_end:
                continue
            block = merged.setdefault(interval, {})
            block[key] = block.get(key, 0.0) + float(amount)
    if not merged:
        return {"blocks": [], "total": None, "complete": False}

    totals = {key: 0.0 for _, key in AMOUNT_LAYERS}
    blocks = []
    for interval in sorted(merged):
        amounts = merged[interval]
        for key, mm in amounts.items():
            totals[key] += mm
        shown = {key: inches(mm) for key, mm in amounts.items() if inches(mm) > 0}
        if shown:
            blocks.append({"when": span(*interval, zone), **shown})

    total = {"precip_in": inches(totals["precip_in"])}
    for key in ("snow_in", "ice_in"):
        if inches(totals[key]) > 0:
            total[key] = inches(totals[key])
    last_end = max(end for _, end in merged)
    total["from"] = clock(min(start for start, _ in merged), zone)
    total["to"] = clock(last_end, zone)
    # The NWS publishes amounts only ~3 days out, so a week-long window usually outruns them. Say so, or a
    # model may present a 3-day total as the week's.
    return {"blocks": blocks, "total": total, "complete": last_end >= window_end}


# ------------------------------------------------------------------------------------------ the tool

def daily_entries(periods):
    """12-hour periods, reduced to the fields worth sending."""
    out = []
    for period in periods:
        rain = (period.get("probabilityOfPrecipitation") or {}).get("value")
        out.append({
            "name": period.get("name"),
            "temperature": f"{period.get('temperature')}°{period.get('temperatureUnit', 'F')}",
            "chance_of_precipitation": f"{rain}%" if rain is not None else None,
            "wind": f"{period.get('windSpeed', '')} {period.get('windDirection', '')}".strip(),
            "forecast": period.get("shortForecast"),
            "details": period.get("detailedForecast"),
        })
    return out


def hourly_entries(periods, zone):
    """Hourly periods, compact (an hourly answer can hold 48 of these)."""
    out = []
    for period in periods:
        rain = (period.get("probabilityOfPrecipitation") or {}).get("value")
        humidity = (period.get("relativeHumidity") or {}).get("value")
        out.append({
            "time": clock(datetime.fromisoformat(period["startTime"]), zone),
            "temperature": f"{period.get('temperature')}°{period.get('temperatureUnit', 'F')}",
            "chance_of_precipitation": f"{rain}%" if rain is not None else None,
            "humidity": f"{humidity}%" if humidity is not None else None,
            "wind": f"{period.get('windSpeed', '')} {period.get('windDirection', '')}".strip(),
            "forecast": period.get("shortForecast"),
        })
    return out


def window_of(periods):
    """
    The span the returned periods cover: start of the first to end of the last.

    Raises:
        ToolError: if there are no periods or their times are unreadable.
    """
    if not periods:
        raise ToolError("the weather service returned no forecast periods")
    try:
        return datetime.fromisoformat(periods[0]["startTime"]), datetime.fromisoformat(periods[-1]["endTime"])
    except (KeyError, TypeError, ValueError):
        raise ToolError("the weather service returned forecast periods without readable times")


def amounts_phrase(total):
    """ "0.09 in precip" (plus snow/ice when forecast) for the one-line summary."""
    parts = [f"{total['precip_in']:.2f} in precip"]
    parts += [f"{total[key]:.1f} in {name}" for key, name in (("snow_in", "snow"), ("ice_in", "ice")) if key in total]
    return ", ".join(parts)


def handle(arguments, config):
    """The tool: resolve the place, find its NWS grid, fetch the forecast and the amounts, keep the useful fields."""
    lat, lon, how = resolve_location(arguments, config)
    detail = str(arguments.get("detail") or "daily").strip().lower()
    if detail not in DETAILS:
        raise ToolError(f"detail must be one of: {', '.join(DETAILS)}")
    hourly = detail == "hourly"
    count = (bounded_int(arguments, "hours", DEFAULT_HOURS, MAX_HOURS) if hourly
             else bounded_int(arguments, "periods", DEFAULT_PERIODS, MAX_PERIODS))

    # NWS asks for at most four decimal places; more gets a redirect.
    point = nws_get(f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}", config)
    properties = point.get("properties") or {}
    place = ((properties.get("relativeLocation") or {}).get("properties") or {})

    forecast = nws_get(nws_url(properties, "forecastHourly" if hourly else "forecast"), config)
    periods = ((forecast.get("properties") or {}).get("periods") or [])[:count]
    window_start, window_end = window_of(periods)
    zone = local_zone(properties.get("timeZone"), window_start.tzinfo)
    entries = hourly_entries(periods, zone) if hourly else daily_entries(periods)

    # The amounts are a second, separate request; a failure there costs the amounts, not the forecast.
    note = ("Forecast amounts in inches, in the NWS's own blocks (amounts are not forecast hour by hour). "
            "precip_in is liquid equivalent (rain plus melted snow/ice). Only blocks with some amount are listed.")
    try:
        grid = nws_get(nws_url(properties, "forecastGridData"), config, timeout=GRID_HTTP_TIMEOUT)
        amounts = precipitation_blocks(grid.get("properties") or {}, window_start, window_end, zone)
        if amounts["total"] is None:
            note = "no precipitation amounts were published for this window"
        elif not amounts["complete"]:
            note += (f" The NWS only publishes amounts through {amounts['total']['to']}; later days have no amount "
                     f"forecast, so the total covers only up to then - not the whole forecast.")
    except ToolError as e:
        amounts, note = {"blocks": [], "total": None, "complete": False}, f"precipitation amounts unavailable: {e}"

    location = f"{place.get('city')}, {place.get('state')}" if place.get("city") else how
    result = {
        "location": location,
        "requested_as": how,
        "source": "National Weather Service (api.weather.gov)",
        "detail": detail,
        ("hours" if hourly else "periods"): entries,
        "precipitation_amounts": {"note": note, "blocks": amounts["blocks"], "total": amounts["total"]},
    }

    # The one line kept in chat history for later turns. The registry cuts it at 160 characters from the END, so the
    # parts go in order of importance: the gist, then the amount total, then (daily) the second period.
    parts = []
    if hourly:
        temps = [t for t in (leading_int(p["temperature"]) for p in entries) if t is not None]
        chances = [c for c in (leading_int(p["chance_of_precipitation"]) for p in entries) if c is not None]
        gist = f"hourly {span(window_start, window_end, zone)}"
        if temps:
            gist += f", {min(temps)}-{max(temps)}°F"
        if chances:
            gist += f", up to {max(chances)}% chance"
        parts.append(gist)
    else:
        parts += [f"{p['name']} {p['forecast']}, {p['temperature']}"
                  + (f", {p['chance_of_precipitation']} rain" if p["chance_of_precipitation"] else "")
                  for p in entries[:2]]
    if amounts["total"]:
        total = amounts["total"]
        parts.insert(1, f"{amounts_phrase(total)} {total['from']}-{total['to']}")
    return ToolAnswer(result, f"{location}: {'; '.join(parts)}")


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
