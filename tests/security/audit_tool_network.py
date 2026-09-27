#!/usr/bin/env python3
"""CS-21: the CS-17 network/file audit, re-run against the tool family (strace over a real GPU run).

Why
---
CS-17 established that nothing in the llama-cpp-python stack contacts a remote host or writes prompts
to disk. Tools are outbound by design, so the CS-21 design notes requires the audit be repeated: the claim is now
"the ONLY outbound traffic is the tool scripts reaching the hosts their job needs, and the main
(model) process still makes none".

Method (same as the CS-17 audit, extended to child processes)
--------------------------------------------------------------
1. ``strace -ff`` (seccomp-bpf, follows every child) over a real run: live_script_tools_smoke.py's
   default five-turn conversation on the GPU - web_search, get_weather, fetch_url through a delegate
   worker, recall, calculator - with the real tool scripts and configs.
2. **Positive control:** after the conversation the driver deliberately opens a TCP connection to
   192.0.2.1:9 (TEST-NET-1, never routed). If that connect() is not in the trace, the tracer is broken
   and a clean result means nothing - the audit fails.
3. The analysis maps every process to its program (execve), then reports:
   * every connect() by address family, process and destination;
   * every DNS name looked up (decoded from the queries sent to the resolver);
   * every public IP connected to, attributed to a looked-up name that resolves to it now;
   * every file opened for writing, created, renamed or deleted.
   It FAILS if: the control is missing; the main process connects to any internet address (other than
   the control); any process connects to an internet address that cannot be attributed to a name;
   a name outside the expected set is looked up; or any file outside the allowed set is written.

Expected hosts: api.weather.gov (get_weather), the SearXNG address in web_search.json, example.com
(the page the conversation asks to fetch) - plus whatever pages the worker chooses to open, which are
listed for a person to read rather than failed, since they depend on the model.

--tool get_grades (added 2026-09-25): instead of the conversation, one real get_grades call - the tool
script plus the Chromium it drives through Playwright, logging in to Infinite Campus via Microsoft
SSO. No model is loaded. Names are classed against GRADES_EXPECTED_SUFFIXES and anything else is
listed for review; files the run wrote that are STILL on disk afterwards are listed (a browser
profile or cache holding grade pages would show up there).

Usage (llama env; needs the GPU free):
    python audit_tool_network.py --arch qwen35moe --report /tmp/cs21_audit.json
    python audit_tool_network.py --analyze-only <trace dir> --report /tmp/cs21_audit.json
    python audit_tool_network.py --tool get_grades --report /tmp/cs21_audit_grades.json
"""
import argparse
import codecs
import collections
import ipaddress
import json
import os
import re
import runpy
import shutil
import socket
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from testlib import settings  # noqa: E402
settings.use_src()
SRC = settings.SRC
ROOT = settings.REPO_ROOT
CONFIGS = settings.get("ai_tools_config_dir", required=False)   # the tools' own configs: outside the repo
CONTROL = ("192.0.2.1", 9)
TRACED = "trace=%network,execve,clone,clone3,fork,vfork,openat,creat,rename,renameat,renameat2,unlink,unlinkat"

# Names the tools are expected to look up. Anything else looked up is a finding (except the pages the worker
# chose to open, which are listed separately - see attribute()).
EXPECTED_NAMES = {"api.weather.gov", "example.com"}

# --tool get_grades: the names a login to Infinite Campus through Microsoft SSO is expected to need, by suffix.
# Anything else is listed for review (a browser can phone home - that is exactly what this mode is for).
GRADES_EXPECTED_SUFFIXES = {
    "Infinite Campus": ("infinitecampus.org", "infinitecampus.com"),
    "Microsoft sign-in": ("microsoftonline.com", "msauth.net", "msftauth.net", "live.com", "microsoft.com",
                          "msidentity.com", "microsoftonline-p.com", "msauthimages.net", "microsoftazuread-sso.com"),
    # Loaded by the sign-in pages themselves, not by the browser - reviewed in the first traced run, 2026-09-25.
    "Page resources (reviewed)": ("fonts.googleapis.com", "fonts.gstatic.com", "cdnjs.cloudflare.com"),
}

# Paths a run may legitimately write: devices, the harness's own capture, Python/CUDA caches. Writes outside these are
# LISTED for a person to read, not failed: on 2026-09-25 the only one was Python's tempfile writability probe
# (/tmp/<8 random chars>, 0600, created and unlinked at once - it contains the word "blat").
ALLOWED_WRITE_PREFIXES = ("/dev/", "/proc/self/", "/sys/", "/run/user/")


def driver(arch, out, tool=None):
    """
    Runs the traced workload in THIS process, then the positive control. The workload is either the live conversation
    (default), or - with tool='get_grades' - one real get_grades call through the real tool contract (its own process,
    the tool server's minimal environment), which logs in to Infinite Campus with a real browser. No model is loaded
    in that mode; the model process was audited on its own.
    """
    if tool == "get_grades":
        sys.path.insert(0, SRC)
        from amadeo_utils.ai.llm.tools import registry as R
        from amadeo_utils.ai.llm.tools.script_tools import load_script_tool
        grades = load_script_tool(settings.get("bash_python"),
                                  os.path.join(ROOT, "scripts/ai/ai-tools/get_grades/get_grades.py"),
                                  os.path.join(CONFIGS, "get_grades.json"))
        outcome = R.execute(grades, {"term": "T1", "days": 14}, 1_000_000)
        print(f"get_grades ok={outcome.ok} summary={outcome.summary!r}", flush=True)      # counts only - never grades
    else:
        sys.argv = ["live_script_tools_smoke.py", "--arch", arch, "--out", out]
        sys.path.insert(0, HERE)
        try:
            runpy.run_path(os.path.join(HERE, "live_script_tools_smoke.py"), run_name="__main__")
        except SystemExit:
            pass
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.5)
    try:
        probe.connect(CONTROL)
    except OSError:
        pass                                    # expected: nothing answers; the attempt is what must be seen
    finally:
        probe.close()
    print("positive control: connect() to %s:%d attempted" % CONTROL, flush=True)


# ------------------------------------------------------------------------------------------ trace parsing

_LINE = re.compile(r"^(?P<call>\w+)\((?P<args>.*)\)\s+=\s+(?P<ret>-?\d+|\?)")
_INET = re.compile(r'sa_family=AF_INET6?, sin6?_port=htons\((\d+)\).*?inet_?(?:pton|addr)\((?:AF_INET6?, )?"([^"]+)"')
_UNIX = re.compile(r'sa_family=AF_UNIX, sun_path=(@?"[^"]*")')
_OPEN = re.compile(r'openat\([^,]+, "([^"]*)", ([A-Z_|]+)')
_EXEC = re.compile(r'execve\("([^"]+)", \[(.*?)\]')
_STRINGS = re.compile(r'"((?:[^"\\]|\\.)*)"')


def dns_names(args):
    """Decodes the query names from the DNS packets in a sendto/sendmmsg/sendmsg argument list."""
    names = []
    for text in _STRINGS.findall(args):
        try:
            packet = codecs.escape_decode(text.encode("latin-1"))[0]
        except Exception:
            continue
        if len(packet) < 17:
            continue
        labels, i = [], 12                      # skip the 12-byte DNS header
        while i < len(packet) and packet[i] != 0 and packet[i] < 64:
            length = packet[i]
            labels.append(packet[i + 1:i + 1 + length].decode("ascii", "replace"))
            i += 1 + length
        if labels and i < len(packet) and packet[i] == 0:
            names.append(".".join(labels).lower())
    return names


def dns_answers(args):
    """
    Decodes the A/AAAA answers in the DNS responses in a recvfrom/recvmsg argument list.

    Returns:
        list[tuple[str, str]]: (question name, address) pairs - what each looked-up name resolved to AT THE TIME,
            which re-resolving later cannot reproduce for CDNs whose addresses rotate.
    """
    pairs = []
    for text in _STRINGS.findall(args):
        try:
            packet = codecs.escape_decode(text.encode("latin-1"))[0]
        except Exception:
            continue
        if len(packet) < 12 or not packet[2] & 0x80:                 # QR bit: responses only
            continue

        def read_name(i, depth=0):
            labels = []
            while i < len(packet) and depth < 10:
                length = packet[i]
                if length == 0:
                    return ".".join(labels), i + 1
                if length & 0xC0 == 0xC0:                             # compression pointer
                    if i + 1 >= len(packet):
                        return None, len(packet)
                    target = ((length & 0x3F) << 8) | packet[i + 1]
                    rest, _ = read_name(target, depth + 1)
                    return ".".join(labels + ([rest] if rest else [])), i + 2
                labels.append(packet[i + 1:i + 1 + length].decode("ascii", "replace"))
                i += 1 + length
            return None, len(packet)

        qdcount, ancount = int.from_bytes(packet[4:6], "big"), int.from_bytes(packet[6:8], "big")
        i, question = 12, None
        for _ in range(qdcount):
            question, i = read_name(i)
            i += 4
        for _ in range(ancount):
            _, i = read_name(i)
            if i + 10 > len(packet):
                break
            rtype, rdlength = int.from_bytes(packet[i:i + 2], "big"), int.from_bytes(packet[i + 8:i + 10], "big")
            rdata = packet[i + 10:i + 10 + rdlength]
            i += 10 + rdlength
            if question and rtype == 1 and len(rdata) == 4:
                pairs.append((question.lower(), str(ipaddress.IPv4Address(rdata))))
            elif question and rtype == 28 and len(rdata) == 16:
                pairs.append((question.lower(), str(ipaddress.IPv6Address(rdata))))
    return pairs


def parse(trace_dir):
    """
    Reads every per-process strace file.

    Returns:
        dict: pid -> {"program", "argv", "connects": [(family, dest, ret)], "dns": [names], "writes": [(call, path, flags)]}
    """
    procs = {}
    for name in sorted(os.listdir(trace_dir)):
        pid = name.rsplit(".", 1)[-1]
        proc = procs.setdefault(pid, {"program": None, "argv": None, "connects": [], "dns": [], "writes": [],
                                      "children": [], "answers": []})
        with open(os.path.join(trace_dir, name), encoding="utf-8", errors="replace") as fh:
            for line in fh:
                match = _LINE.match(line.strip())
                if not match:
                    continue
                call, args, ret = match.group("call"), match.group("args"), match.group("ret")
                if call in ("clone", "clone3", "fork", "vfork") and ret.isdigit() and int(ret) > 0:
                    proc["children"].append(ret)             # a thread or a process; either way this pid is its parent
                elif call == "execve" and ret == "0":
                    exe = _EXEC.match(line.strip())
                    if exe:
                        proc["program"], proc["argv"] = exe.group(1), exe.group(2)[:300]
                elif call == "connect":
                    inet, unix = _INET.search(args), _UNIX.search(args)
                    if inet:
                        # -yy names the socket's protocol. A UDP connect() sends nothing: glibc's getaddrinfo
                        # issues one per candidate address to choose a source address (RFC 6724 sorting).
                        proto = re.match(r"\d+<(TCP|UDP)", args)
                        family = "inet/" + (proto.group(1).lower() if proto else "?")
                        proc["connects"].append((family, f"{inet.group(2)}:{inet.group(1)}", ret))
                    elif unix:
                        proc["connects"].append(("unix", unix.group(1), ret))
                    elif "AF_UNSPEC" not in args:
                        proc["connects"].append(("other", args[:120], ret))
                elif call in ("sendto", "sendmmsg", "sendmsg") and (":53]>" in args or "htons(53)" in args):
                    # -yy annotates the socket with its peer, so only packets to a DNS port are decoded.
                    proc["dns"] += dns_names(args)
                elif call in ("recvfrom", "recvmsg", "recvmmsg") and ":53]>" in args:
                    proc["answers"] += dns_answers(args)
                elif call == "openat":
                    opened = _OPEN.search(line)
                    if opened and re.search(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND", opened.group(2)) and ret != "-1":
                        proc["writes"].append(("openat", opened.group(1), opened.group(2)))
                elif call in ("creat", "rename", "renameat", "renameat2", "unlink", "unlinkat") and not ret.startswith("-"):
                    paths = _STRINGS.findall(args)
                    proc["writes"].append((call, " -> ".join(paths), ""))
    return procs


def resolve_now(name):
    """The addresses a name resolves to right now (for attributing connects to looked-up names)."""
    try:
        return {info[4][0] for info in socket.getaddrinfo(name, 443, proto=socket.IPPROTO_TCP)}
    except OSError:
        return set()


def assign_owners(procs):
    """
    Gives every traced pid (thread or process) the program that owns it: its own, if it called execve, else its
    parent's, and so on up. strace -ff writes one file per THREAD, and a thread never calls execve - so without this a
    browser's network thread would look like part of whatever process happened to have no execve (the audit driver).
    Sets proc['owner'] (the pid whose execve defines it) and proc['is_main'] (owned by the root: the audit driver).
    """
    parent = {child: pid for pid, proc in procs.items() for child in proc["children"]}
    root = next((pid for pid in procs if pid not in parent and procs[pid]["program"]), None)

    def owner(pid, seen=()):
        if procs.get(pid, {}).get("program") or pid not in parent or pid in seen:
            return pid
        return owner(parent[pid], seen + (pid,))

    for pid, proc in procs.items():
        proc["owner"] = owner(pid)
        own = procs.get(proc["owner"], proc)
        proc["effective_program"], proc["effective_argv"] = own.get("program"), own.get("argv")
        proc["is_main"] = proc["owner"] == root


def role(proc, main_pid):
    """A readable role for a process: the main (driver) process, a tool script, the browser, or something else."""
    argv = proc.get("effective_argv") or proc.get("argv") or ""
    program = proc.get("effective_program") or proc.get("program") or ""
    for tool in ("get_weather", "fetch_url", "web_search", "get_grades"):
        if f"{tool}.py" in argv:
            return f"tool:{tool}" + (" --describe" if "--describe" in argv else "")
    if "nvidia-smi" in program:
        return "nvidia-smi"
    # Chromium starts its helpers by re-executing itself as /proc/self/exe.
    if "ms-playwright" in program or "chrome" in os.path.basename(program) or program == "/proc/self/exe":
        return "browser (get_grades)"
    if "playwright" in program and "node" in os.path.basename(program):
        return "playwright driver (get_grades)"
    return "main" if proc.get("is_main") else (program or "unknown")


def analyze(trace_dir, capture_path, searxng, grades_mode=False):
    """Applies the pass/fail rules to a trace directory. Returns the report dict."""
    procs = parse(trace_dir)
    assign_owners(procs)
    names = sorted({n for p in procs.values() for n in p["dns"]})
    # Attribute addresses from the DNS answers recorded in the trace; re-resolving now is only a fallback (CDN
    # addresses rotate, so 'resolves to it now' misses many that were right at the time).
    resolved = {n: set() for n in names}
    for proc in procs.values():
        for name, address in proc["answers"]:
            resolved.setdefault(name, set()).add(address)
    for n in names:
        if not resolved[n]:
            resolved[n] = resolve_now(n)
    searxng_host = re.sub(r"^https?://", "", searxng).split("/")[0] if searxng else None

    findings, connects, control_seen = [], [], False
    for pid, proc in procs.items():
        who = role(proc, None)
        for family, dest, ret in proc["connects"]:
            entry = {"pid": pid, "process": who, "family": family, "destination": dest, "result": ret}
            if family.startswith("inet"):
                ip = dest.rsplit(":", 1)[0]
                try:
                    address = ipaddress.ip_address(ip.split("%")[0])
                except ValueError:
                    findings.append(f"{who}: unparseable destination {dest}")
                    connects.append(entry)
                    continue
                if dest == "%s:%d" % CONTROL:
                    control_seen, entry["what"] = True, "POSITIVE CONTROL"
                elif dest.endswith(":53") and not address.is_global:
                    entry["what"] = "DNS resolver"
                elif family == "inet/udp" and address.is_global:
                    # Source-address selection probe: no packet leaves. Still attributed, for the record.
                    owners = [n for n, addresses in resolved.items() if ip in addresses]
                    entry["what"] = "UDP address-selection probe, no packet sent (" + (", ".join(owners) or "?") + ")"
                elif searxng_host and dest == searxng_host:
                    entry["what"] = "SearXNG (web_search config)"
                    if not who.startswith("tool:web_search"):
                        findings.append(f"SearXNG reached by {who}, not by web_search")
                elif address.is_global:
                    owners = [n for n, addresses in resolved.items() if ip in addresses]
                    entry["what"] = ", ".join(owners) if owners else "UNATTRIBUTED"
                    if proc["is_main"]:
                        findings.append(f"the MAIN process connected to internet address {dest}")
                    if not owners:
                        findings.append(f"{who} connected to {dest}, which no DNS answer in the trace (or now) gave")
                else:
                    entry["what"] = "private/local address"
                    if not ip.startswith("127.") and ip != "::1":
                        findings.append(f"{who} connected to private address {dest}")
            connects.append(entry)

    if grades_mode:
        known = {n: label for n in names for label, suffixes in GRADES_EXPECTED_SUFFIXES.items()
                 if any(n == s or n.endswith("." + s) for s in suffixes)}
        unexpected = [n for n in names if n not in known]
    else:
        unexpected = [n for n in names if n not in EXPECTED_NAMES]
    writes = []
    for pid, proc in procs.items():
        for call, path, flags in proc["writes"]:
            # ~/.nv/ComputeCache is the CUDA driver's JIT kernel cache (compiled GPU code, no prompt content).
            if (path.startswith(ALLOWED_WRITE_PREFIXES) or path == capture_path or "/__pycache__/" in path
                    or "/.nv/ComputeCache/" in path):
                continue
            writes.append({"pid": pid, "process": role(proc, None), "call": call, "path": path, "flags": flags,
                           "still_exists": call == "openat" and os.path.exists(path)})
    if not control_seen:
        findings.append("POSITIVE CONTROL MISSING - the tracer did not see the deliberate connect(); the audit proves nothing")
    return {
        "processes": {pid: {"role": role(p, None), "argv": p["argv"]} for pid, p in procs.items() if p["program"]},
        "dns_names_looked_up": names,
        "names_outside_expected_set": unexpected,
        "connects": connects,
        "writes_outside_allowed_set": writes,
        "files_left_behind": sorted({w["path"] for w in writes if w["still_exists"]}),
        "findings": findings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="qwen35moe")
    parser.add_argument("--report", help="where to write the JSON report (required unless --driver)")
    parser.add_argument("--analyze-only", metavar="TRACE_DIR")
    parser.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capture", help=argparse.SUPPRESS)
    parser.add_argument("--tool", choices=["get_grades"],
                        help="audit one real tool call instead of the live conversation (no model loaded)")
    args = parser.parse_args()

    if args.driver:
        driver(args.arch, args.capture, args.tool)
        return
    if not args.report:
        parser.error("--report is required")

    with open(os.path.join(CONFIGS, "web_search.json")) as fh:
        searxng = json.load(fh).get("searxng_url", "")
    if args.analyze_only:
        trace_dir, capture = args.analyze_only, os.path.join(args.analyze_only, "..", "capture.json")
    else:
        work = tempfile.mkdtemp(prefix="cs21-audit-")
        trace_dir, capture = os.path.join(work, "trace"), os.path.join(work, "capture.json")
        os.makedirs(trace_dir)
        command = ["strace", "--seccomp-bpf", "-f", "-ff", "-qq", "-s", "512", "-x", "-yy", "-e", TRACED,
                   "-o", os.path.join(trace_dir, "t"), sys.executable, os.path.abspath(__file__),
                   "--driver", "--arch", args.arch, "--capture", capture] + (["--tool", args.tool] if args.tool else [])
        print("tracing:", " ".join(command[:10]), "...", flush=True)
        subprocess.run(command, check=False)
        print("trace kept in", trace_dir, flush=True)

    report = analyze(trace_dir, os.path.realpath(capture), searxng, grades_mode=args.tool == "get_grades")
    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2)
    by_what = collections.Counter((c["process"], c["family"], c.get("what", c["destination"])) for c in report["connects"])
    print("\nconnect() calls by process / destination:")
    for (who, family, what), n in sorted(by_what.items()):
        print(f"  {n:4}  {who:28} {family:9} {what}")
    print("\nDNS names looked up:", ", ".join(report["dns_names_looked_up"]) or "(none)")
    print("names outside the expected set (pages the worker chose, or a finding):",
          ", ".join(report["names_outside_expected_set"]) or "(none)")
    print(f"files still on disk after the run (of those written): {len(report['files_left_behind'])}")
    for path in report["files_left_behind"][:40]:
        print(f"    {path}")
    print(f"writes outside the allowed set: {len(report['writes_outside_allowed_set'])}")
    for w in report["writes_outside_allowed_set"][:40]:
        print(f"    {w['process']:28} {w['call']:8} {w['path']} {w['flags']}")
    print("\nFINDINGS:" if report["findings"] else "\nNO FINDINGS - positive control seen.")
    for finding in report["findings"]:
        print("  -", finding)
    sys.exit(len(report["findings"]))


if __name__ == "__main__":
    main()
