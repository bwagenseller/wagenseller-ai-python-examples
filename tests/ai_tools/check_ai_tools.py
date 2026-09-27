#!/usr/bin/env python3
"""CS-21: the tool-script contract and the first two script tools, get_weather and fetch_url.

What it proves
--------------
* Contract (script_tools.py), with a probe script: the tool's environment holds only PATH, HOME,
  LANG/LC_ALL/TZ and PYTHONPATH (a secret in the server's environment does not reach it); the
  arguments are not on the command line; a tool that overruns its timeout is KILLED, not just
  abandoned; a clean failure's message comes through; a crash is reported without its traceback
  (which may contain argument values).
* Both real tools describe themselves with the right flags.
* fetch_url's network guard: every private / local address range is refused, including behind a
  redirect from an allowed host; allow_private_hosts works; non-text content and oversize pages
  are handled. DNS rebinding is closed: the name is resolved once per hop and the connection is
  pinned to the checked address (simulated with a rebinding resolver); the Host header still
  carries the URL's name; certificates are still verified against the name (live, badssl.com).
* get_weather's amount handling, offline on synthetic gridpoint data: ISO 8601 interval parsing,
  mm -> inches, 6-hour blocks kept whole where they straddle the window edges, zero blocks dropped
  from the list but counted in the total, snow/ice only when forecast, non-mm layers refused.
* LIVE (small, benign): NWS forecasts by ZIP, by coordinates and for home (daily and hourly, with
  amounts; hours capped at 48; bad detail refused), and https://example.com.

Usage (in the agent-tools env):
    PYTHONPATH=<repo>/src <agent-tools python> check_ai_tools.py [--offline]
"""
import argparse
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
ROOT = settings.REPO_ROOT
from amadeo_utils.ai.llm.tools import registry as R                  # noqa: E402
from amadeo_utils.ai.llm.tools.script_tools import load_script_tool   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
WEATHER = os.path.join(ROOT, "scripts", "ai", "ai-tools", "get_weather", "get_weather.py")
FETCH = os.path.join(ROOT, "scripts", "ai", "ai-tools", "fetch_url", "fetch_url.py")
SEARCH = os.path.join(ROOT, "scripts", "ai", "ai-tools", "web_search", "web_search.py")
CONFIGS = settings.get("ai_tools_config_dir", required=False)   # the tools' own configs: outside the repo
failures = []


def check(ok, label, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"\n        {detail}" if not ok and detail else ""))
    if not ok:
        failures.append(label)


def contract_checks():
    print("== contract (probe script) ==")
    os.environ["CS21_FAKE_TOKEN"] = "must-not-leak"
    probe = load_script_tool(PYTHON, os.path.join(HERE, "tool_script_probe.py"))
    out = R.execute(probe, {"secret_arg": "ARGVALUE"}, 100000)
    data = json.loads(out.content) if out.ok else {}
    check(out.ok and set(data["env"]) <= {"PATH", "HOME", "LANG", "LC_ALL", "TZ", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE"},
          "the tool's environment is only PATH/HOME/LANG/TZ/PYTHONPATH", f"env={data.get('env')}")
    check("CS21_FAKE_TOKEN" not in data.get("env", []), "a secret in the server's environment does not reach the tool")
    check(not any("ARGVALUE" in a for a in data.get("argv", [])), "arguments are not on the command line (ps cannot see them)")

    started = time.monotonic()
    out = R.execute(probe, {"mode": "sleep"}, 1000)
    elapsed = time.monotonic() - started
    time.sleep(0.3)
    leftover = subprocess.run(["pgrep", "-f", "tool_script_probe.py"], capture_output=True, text=True).stdout.strip()
    check(not out.ok and "stopped" in out.content and elapsed < 5, f"an overrunning tool fails at its timeout ({elapsed:.1f}s)",
          out.content)
    check(leftover == "", "an overrunning tool's process is killed, not abandoned", f"still running: {leftover}")

    out = R.execute(probe, {"mode": "fail"}, 1000)
    check(not out.ok and "clean failure message" in out.content, "a clean failure's message comes through", out.content)
    out = R.execute(probe, {"mode": "crash", "secret_arg": "ARGVALUE"}, 1000)
    check(not out.ok and "ARGVALUE" not in out.content, "a crash is reported without its traceback", out.content)


def describe_checks():
    print("== the two tools describe themselves ==")
    weather = load_script_tool(PYTHON, WEATHER, os.path.join(CONFIGS, "get_weather.json"))
    fetch = load_script_tool(PYTHON, FETCH, os.path.join(CONFIGS, "fetch_url.json"))
    check(weather.name == "get_weather" and weather.outbound and not weather.untrusted_output and not weather.private,
          "get_weather: outbound only")
    check(fetch.name == "fetch_url" and fetch.outbound and fetch.untrusted_output and not fetch.private,
          "fetch_url: outbound + untrusted_output")
    search = load_script_tool(PYTHON, SEARCH, os.path.join(CONFIGS, "web_search.json"))
    check(search.name == "web_search" and search.outbound and search.untrusted_output and not search.private,
          "web_search: outbound + untrusted_output")
    return weather, fetch, search


def load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_url_module", FETCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard_checks(offline):
    print("== fetch_url's network guard ==")
    f = load_fetch_module()
    refused = ["file:///etc/passwd", "ftp://example.com/", "http://127.0.0.1/", "http://localhost/", "http://192.168.1.1/",
               "http://10.0.0.1/", "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/",
               "http://[::1]/", "http://0.0.0.0/", "https://user:pw@example.com/", "http:///nohost"]
    for url in refused:
        try:
            f.check_url(url, {})
            check(False, f"refused: {url}", "it was allowed")
        except f.ToolError:
            check(True, f"refused: {url}")
    try:
        f.check_url("http://localhost/", {"allow_private_hosts": ["localhost"]})
        check(True, "allow_private_hosts lets a listed LAN host through")
    except f.ToolError as e:
        check(False, "allow_private_hosts lets a listed LAN host through", str(e))
    if not offline:
        try:
            f.check_url("https://example.com/", {})
            check(True, "a public host is allowed")
        except f.ToolError as e:
            check(False, "a public host is allowed", str(e))

    # A local server: '127.0.0.1' is trusted by config for the first hop; its redirect to 'localhost' is not.
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", f"http://localhost:{self.server.server_port}/text")
                self.end_headers()
            elif self.path == "/image":
                self.send_response(200); self.send_header("Content-Type", "image/png"); self.end_headers()
                self.wfile.write(b"\x89PNG....")
            elif self.path == "/host":
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
                self.wfile.write(("host=" + str(self.headers.get("Host"))).encode())
            elif self.path == "/big":
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
                self.wfile.write(b"x" * 50000)
            else:
                self.send_response(200); self.send_header("Content-Type", "text/plain"); self.end_headers()
                self.wfile.write(b"hello")

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base, trust = f"http://127.0.0.1:{server.server_port}", {"allow_private_hosts": ["127.0.0.1"], "max_bytes": 1000}
    try:
        f.handle({"url": base + "/redirect"}, trust)
        check(False, "a redirect from an allowed host to a private one is refused", "it was followed")
    except f.ToolError as e:
        check("private" in str(e), "a redirect from an allowed host to a private one is refused", str(e))
    try:
        f.handle({"url": base + "/image"}, trust)
        check(False, "non-text content is refused")
    except f.ToolError as e:
        check("not text" in str(e), "non-text content is refused", str(e))
    out = f.handle({"url": base + "/big"}, trust).result      # handle() returns a ToolAnswer (result + summary)
    check(out["truncated"] and len(out["text"]) == 1000, "a page is cut at max_bytes", f"len={len(out['text'])}")

    # Pinning (DNS rebinding closed 2026-09-25): the connection goes to an IP, but the Host header keeps the name.
    port = server.server_port
    out = f.handle({"url": f"http://localhost:{port}/host"}, {"allow_private_hosts": ["localhost"]}).result
    check(out["text"] == f"host=localhost:{port}", "a pinned connection still sends the URL's own name as Host",
          out["text"])

    # A rebinding resolver: the first answer for 'rebind.test' is a public address, every later one is the local
    # server. The fetch must resolve once and connect to the address that was checked. make_pool is wrapped to
    # record where the connection goes and stop there, so nothing leaves this machine.
    real_getaddrinfo, calls, pinned_to = socket.getaddrinfo, [], []

    def rebinding(host, port_, *args, **kwargs):
        if host != "rebind.test":
            return real_getaddrinfo(host, port_, *args, **kwargs)
        calls.append(host)
        address = "8.8.8.8" if len(calls) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port_))]

    def recording_pool(scheme, address, port_, host):
        pinned_to.append(address)
        raise f.ToolError("stopped by the test before connecting")

    # Fallback: a name whose first checked address is unreachable (::1 - the test server is IPv4 only) must be
    # fetched from its next checked address, not failed and not re-resolved.
    def two_addresses(host, port_, *args, **kwargs):
        if host != "fallback.test":
            return real_getaddrinfo(host, port_, *args, **kwargs)
        return [(socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", port_, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port_))]

    socket.getaddrinfo = two_addresses
    try:
        out = f.handle({"url": f"http://fallback.test:{port}/host"}, {"allow_private_hosts": ["fallback.test"]}).result
        fell_back = out["text"]
    except f.ToolError as e:
        fell_back = f"failed: {e}"
    finally:
        socket.getaddrinfo = real_getaddrinfo
    check(fell_back == f"host=fallback.test:{port}", "an unreachable checked address falls through to the next one",
          fell_back)

    socket.getaddrinfo, real_make_pool, f.make_pool = rebinding, f.make_pool, recording_pool
    try:
        try:
            f.handle({"url": f"http://rebind.test:{port}/text"}, {})
            reached = True
        except f.ToolError:
            reached = False
    finally:
        socket.getaddrinfo, f.make_pool = real_getaddrinfo, real_make_pool
    check(not reached and pinned_to == ["8.8.8.8"] and len(calls) == 1,
          "DNS rebinding: the name is resolved once and the connection goes to the address that was checked",
          f"reached={reached} pinned_to={pinned_to} resolutions={len(calls)}")
    server.shutdown()


def search_checks(search, offline):
    print("== web_search ==")
    out = R.execute(search, {"query": "   "}, 100000)
    check(not out.ok and "query is required" in out.content, "an empty query fails cleanly", out.content)
    out = R.execute(search, {"query": "x", "time_range": "decade"}, 100000)
    check(not out.ok and "time_range" in out.content, "an invalid time_range fails cleanly", out.content)
    # Pointed at a port nothing listens on: must fail cleanly, not hang or crash.
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"searxng_url": "http://127.0.0.1:9"}, fh)
    dead = load_script_tool(PYTHON, SEARCH, fh.name)
    out = R.execute(dead, {"query": "test"}, 100000)
    os.unlink(fh.name)
    check(not out.ok and "could not be reached" in out.content, "an unreachable SearXNG fails cleanly", out.content)
    engine_outage_checks()
    if offline:
        return
    out = R.execute(search, {"query": "national weather service api documentation", "max_results": 3}, 100000)
    data = json.loads(out.content) if out.ok else {}
    results = data.get("results", [])
    check(out.ok and len(results) == 3 and all(r["url"].startswith("http") and r["title"] for r in results),
          f"live search returns 3 results: {[r['url'] for r in results]}", out.content[:300])
    check(all(len(r["snippet"]) <= 400 for r in results), "snippets are capped at 400 characters")
    out = R.execute(search, {"query": "weather", "max_results": 50}, 100000)
    data = json.loads(out.content) if out.ok else {}
    check(out.ok and len(data.get("results", [])) <= 10, "max_results is capped at 10 whatever the model asks for",
          f"{len(data.get('results', []))} results")
    out = R.execute(search, {"query": "hurricane", "time_range": "week", "max_results": 2}, 100000)
    check(out.ok, "a time_range search works", out.content[:200])


def engine_outage_checks():
    """
    A stand-in SearXNG on 127.0.0.1 (2026-09-26): an empty search while SearXNG reports engines that did not answer
    must say web search is down, not "no results" - live, "no results" made workers retry against engines that were
    already suspended. A genuinely empty search still says "no results", and results still win over a partial outage.
    """
    canned = {
        "outage": {"results": [], "answers": [], "unresponsive_engines": [["brave", "Suspended: too many requests"],
                                                                            ["duckduckgo", "timeout"]]},
        "empty": {"results": [], "answers": [], "unresponsive_engines": []},
        "partial": {"results": [{"url": "https://example.com/", "title": "Example", "content": "x"}], "answers": [],
                    "unresponsive_engines": [["brave", "Suspended: too many requests"]]},
    }

    class FakeSearx(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("q", [""])[0]
            body = json.dumps(canned[query]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSearx)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"searxng_url": f"http://127.0.0.1:{server.server_address[1]}"}, fh)
    try:
        fake = load_script_tool(PYTHON, SEARCH, fh.name)
        out = R.execute(fake, {"query": "outage"}, 100000)
        check(not out.ok and "temporarily unavailable" in out.content and "brave" not in out.content,
              "engines suspended -> 'temporarily unavailable', naming no engines", out.content)
        out = R.execute(fake, {"query": "empty"}, 100000)
        check(not out.ok and out.content.endswith("no results"), "nothing found, engines fine -> 'no results'", out.content)
        out = R.execute(fake, {"query": "partial"}, 100000)
        check(out.ok and json.loads(out.content)["results"], "results despite a suspended engine -> results", out.content[:200])
    finally:
        server.shutdown()
        os.unlink(fh.name)


def load_weather_module():
    spec = importlib.util.spec_from_file_location("get_weather_module", WEATHER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weather_amount_checks():
    """get_weather's precipitation-amount logic on synthetic gridpoint data - no network."""
    print("== get_weather amounts (offline) ==")
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    w = load_weather_module()
    utc = timezone.utc
    t = lambda h: datetime(2026, 9, 26, 0, tzinfo=utc).replace(hour=h)     # noqa: E731 - hour h on 2026-09-26 UTC

    check(w.parse_valid_time("2026-09-26T06:00:00+00:00/PT6H") == (t(6), t(12)), "parses start/PT6H")
    start, end = w.parse_valid_time("2026-09-26T00:00:00+00:00/P1DT6H")
    check((end - start).total_seconds() == 30 * 3600, "parses start/P1DT6H")
    check(all(w.parse_valid_time(v) is None for v in ("junk", None, "2026-09-26T00:00:00+00:00/PT0H",
                                                      "2026-09-26T00:00:00/PT6H", "2026-09-26T00:00:00+00:00/6H")),
          "rejects junk, zero-length, zone-less and malformed intervals")
    check(w.inches(25.4) == 1.0 and w.inches(2.286) == 0.09, "mm -> inches to the hundredth")

    ny = ZoneInfo("America/New_York")
    grid = {
        "quantitativePrecipitation": {"uom": "wmoUnit:mm", "values": [
            {"validTime": "2026-09-26T00:00:00+00:00/PT6H", "value": 0},        # overlaps the window start, zero
            {"validTime": "2026-09-26T06:00:00+00:00/PT6H", "value": 2.286},
            {"validTime": "2026-09-26T12:00:00+00:00/PT6H", "value": 12.7},     # straddles the window end
            {"validTime": "2026-09-26T18:00:00+00:00/PT6H", "value": 50.0},     # wholly after the window
        ]},
        "snowfallAmount": {"uom": "wmoUnit:mm", "values": [
            {"validTime": "2026-09-26T06:00:00+00:00/PT6H", "value": 0}]},
        "iceAccumulation": {"uom": "wmoUnit:mm", "values": [
            {"validTime": "2026-09-26T12:00:00+00:00/PT6H", "value": 2.54}]},
    }
    out = w.precipitation_blocks(grid, t(3), t(14), ny)
    check([b["when"] for b in out["blocks"]] == ["Sat 2 AM-8 AM", "Sat 8 AM-2 PM"],
          "only blocks with an amount are listed, in local time, straddling blocks kept whole", out["blocks"])
    check(out["blocks"][1] == {"when": "Sat 8 AM-2 PM", "precip_in": 0.5, "ice_in": 0.1},
          "a block carries each non-zero layer", out["blocks"][1])
    check("snow_in" not in out["blocks"][0] and "snow_in" not in out["total"], "zero snow is not mentioned")
    check(out["total"] == {"precip_in": 0.59, "ice_in": 0.1, "from": "Fri 8 PM", "to": "Sat 2 PM"},
          "the total covers exactly the kept blocks, from the first's start to the last's end", out["total"])
    check(out["complete"] is True, "a block straddling the window end counts as covering it")
    late = w.precipitation_blocks(grid, t(3), t(23).replace(day=28), ny)
    check(late["complete"] is False and late["total"]["to"] == "Sat 8 PM",
          "amounts that stop before the window ends are flagged incomplete", late)
    check(w.precipitation_blocks(grid, t(20).replace(day=27), t(22).replace(day=27), ny) == {"blocks": [], "total": None, "complete": False},
          "no block overlapping the window -> no total (not a false 0.00)")
    feet = {"quantitativePrecipitation": {"uom": "wmoUnit:in", "values": grid["quantitativePrecipitation"]["values"]}}
    check(w.precipitation_blocks(feet, t(3), t(14), ny)["total"] is None, "a layer not in mm is refused, not guessed")
    check(w.amounts_phrase({"precip_in": 0.59, "ice_in": 0.1}) == "0.59 in precip, 0.1 in ice", "summary phrase")
    check(w.leading_int("-5°F") == -5 and w.leading_int("30%") == 30 and w.leading_int(None) is None,
          "reads numbers back out of '-5°F' / '30%'")
    worst = 2 * sum(w.HTTP_TIMEOUT) + sum(w.GRID_HTTP_TIMEOUT)
    check(w.DEFINITION["timeout_s"] >= worst and w.GRID_HTTP_TIMEOUT[1] > w.HTTP_TIMEOUT[1],
          f"the tool's timeout ({w.DEFINITION['timeout_s']} s) covers all three requests at their limits ({worst} s), "
          f"so a slow gridpoint costs only the amounts")


def live_checks(weather, fetch):
    print("== live (NWS, example.com) ==")
    out = R.execute(weather, {"zip": "10001", "periods": 2}, 100000)
    data = json.loads(out.content) if out.ok else {}
    check(out.ok and len(data.get("periods", [])) == 2 and "New York" in data.get("location", ""),
          f"get_weather by ZIP: {data.get('location')} - {data.get('periods', [{}])[0].get('forecast')}", out.content[:300])
    out = R.execute(weather, {"latitude": 38.8977, "longitude": -77.0365, "periods": 1}, 100000)
    check(out.ok, "get_weather by coordinates", out.content[:300])
    out = R.execute(weather, {"zip": "00000"}, 100000)
    check(not out.ok and "unknown US ZIP" in out.content, "get_weather: an unknown ZIP fails cleanly", out.content)
    out = R.execute(weather, {"latitude": 51.5, "longitude": -0.12}, 100000)
    check(not out.ok and "US only" in out.content, "get_weather: outside the US fails cleanly", out.content)
    with open(os.path.join(CONFIGS, "get_weather.json")) as f:
        homeless = {k: v for k, v in json.load(f).items() if not k.startswith("home_")}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(homeless, f)
    try:
        out = R.execute(load_script_tool(PYTHON, WEATHER, f.name), {}, 100000)
    finally:
        os.unlink(f.name)
    check(not out.ok and "no home location" in out.content,
          "get_weather: no location and no home configured fails cleanly", out.content)

    out = R.execute(weather, {}, 100000)
    data = json.loads(out.content) if out.ok else {}
    amounts = data.get("precipitation_amounts", {})
    check(out.ok and data.get("requested_as") == "Home" and data.get("detail") == "daily" and len(data.get("periods", [])) == 4,
          f"get_weather for home, daily by default: {data.get('location')}", out.content[:300])
    check(out.ok and amounts.get("total") and "precip_in" in amounts["total"] and isinstance(amounts.get("blocks"), list),
          f"daily includes an amount total: {amounts.get('total')}", json.dumps(amounts)[:300])
    out = R.execute(weather, {"periods": 14}, 1000000)
    note = (json.loads(out.content).get("precipitation_amounts") or {}).get("note", "") if out.ok else ""
    check(out.ok and "only publishes amounts through" in note,
          "a 7-day forecast says the amounts stop early (NWS publishes ~3 days)", note)
    out = R.execute(weather, {"detail": "hourly", "hours": 24}, 100000)
    data = json.loads(out.content) if out.ok else {}
    hours = data.get("hours", [])
    check(out.ok and len(hours) == 24 and all({"time", "temperature", "chance_of_precipitation", "humidity", "wind",
                                                "forecast"} <= set(h) for h in hours),
          f"hourly for 24 hours: {hours[0] if hours else None}", out.content[:300])
    check(out.ok and (data.get("precipitation_amounts") or {}).get("total"),
          f"hourly includes an amount total: {(data.get('precipitation_amounts') or {}).get('total')}")
    check(out.ok and len(out.content) < 3000 * 4, f"24 hours fit the default result cap ({len(out.content)} chars)")
    out = R.execute(weather, {"detail": "hourly", "hours": 500}, 1000000)
    data = json.loads(out.content) if out.ok else {}
    check(out.ok and len(data.get("hours", [])) == 48, "hours is capped at 48 whatever the model asks for")
    check(out.ok and len(out.content) < 3000 * 4, f"48 hours still fit the default result cap ({len(out.content)} chars)")
    out = R.execute(weather, {"detail": "weekly"}, 100000)
    check(not out.ok and "detail must be one of" in out.content, "get_weather: an unknown detail fails cleanly", out.content)
    out = R.execute(fetch, {"url": "https://example.com/"}, 100000)
    data = json.loads(out.content) if out.ok else {}
    check(out.ok and data.get("title") == "Example Domain" and "domain" in data.get("text", "").lower(),
          "fetch_url: https://example.com", out.content[:300])
    check(bool(out.summary) and out.summary.startswith("fetched https://example.com") and "Example Domain" not in out.summary,
          f"fetch_url's history summary quotes nothing from the page: {out.summary!r}")
    # Pinning connects to an IP address; the certificate must still be verified against the NAME. badssl.com
    # serves deliberately broken certificates for exactly this kind of test.
    for url, what in (("https://self-signed.badssl.com/", "a self-signed certificate"),
                      ("https://wrong.host.badssl.com/", "a certificate for the wrong name"),
                      ("https://expired.badssl.com/", "an expired certificate")):
        out = R.execute(fetch, {"url": url}, 100000)
        check(not out.ok and "SSLError" in out.content, f"fetch_url refuses {what}", out.content[:200])
    out = R.execute(fetch, {"url": "https://badssl.com/"}, 100000)
    check(out.ok, "fetch_url accepts badssl.com's valid certificate (the refusals above are not a broken TLS setup)",
          out.content[:200])


def summary_checks(weather, search, offline):
    print("== history summaries ==")
    short = R.summarize_content(json.dumps({"location": "New York, NY", "periods": [1, 2, 3], "extra": {"a": 1}}))
    check(short == "location: New York, NY; periods: 3 items; extra: ...", f"generic summary of a JSON object: {short!r}")
    long = R.shorten("word " * 80)
    check(len(long) <= 160 and long.endswith("…") and not long.endswith(" …"), "long text is cut at a word boundary")
    if offline:
        return
    out = R.execute(weather, {"zip": "10001", "periods": 2}, 100000)
    check(out.ok and out.summary and out.summary.startswith("New York, NY: ") and "°" in out.summary
          and not out.summary.startswith("{") and len(out.summary) <= 160,
          f"get_weather writes its own summary: {out.summary!r}")
    for arguments in ({}, {"detail": "hourly", "hours": 48}):
        out = R.execute(weather, arguments, 1000000)
        check(out.ok and " in precip " in (out.summary or "") and len(out.summary) <= 160,
              f"the amount total survives the summary cut ({arguments or 'daily'}): {out.summary!r}")
    out = R.execute(search, {"query": "national weather service api", "max_results": 3}, 100000)
    check(out.ok and out.summary == "3 results for 'national weather service api'",
          f"web_search's summary names only the query and count: {out.summary!r}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true", help="skip everything that touches the internet")
    args = parser.parse_args()
    contract_checks()
    weather, fetch, search = describe_checks()
    guard_checks(args.offline)
    search_checks(search, args.offline)
    weather_amount_checks()
    if not args.offline:
        live_checks(weather, fetch)
    summary_checks(weather, search, args.offline)
    print("\nALL CHECKS PASS" if not failures else f"\n{len(failures)} check(s) FAILED")
    sys.exit(len(failures))


if __name__ == "__main__":
    main()
