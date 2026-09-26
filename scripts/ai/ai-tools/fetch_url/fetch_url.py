#!/usr/bin/env python3
"""
fetch_url - fetches one web page and returns its readable text (CS-21 tool).

Runs in the 'agent-tools' conda env as a tool script (see amadeo_utils/ai/llm/tools/script_tools.py):
the tool server calls it with the arguments on stdin and reads one JSON answer from stdout.

Security flags: 'outbound' (the URL, chosen by the model, is fetched) and 'untrusted_output' (the
page is written by anyone). In the tool server's default 'auto' mode that means only a delegate
WORKER can call this, and what it reads never reaches the main model.

What it refuses, and why
------------------------
The model chooses the URL, and a page it read earlier may have chosen it for the model. So:

* Only http and https. No file://, ftp://, and no 'user:password@' in the URL.
* **No addresses on this network, or this machine.** Every address the host name resolves to must
  be a public internet address - not loopback, not 192.168.x.x / 10.x / 172.16-31.x, not
  link-local, not the carrier-grade NAT range. Otherwise injected text could point the worker at
  the router, a NAS, Pi-hole, or a service on this host (server-side request forgery). Redirects
  are followed by hand, at most MAX_REDIRECTS, and every hop is checked the same way - a public
  page redirecting to 192.168.1.1 is refused.
  Hosts the operator trusts anyway can be listed in the config ('allow_private_hosts').
* Pages that are too big: at most 'max_bytes' are downloaded.
* Content that is not text: HTML, plain text, JSON, XML and Markdown only.

The address that is checked is the address that is used (DNS rebinding)
------------------------------------------------------------------------
The host name is resolved exactly ONCE per hop, in check_url, and the connection is then opened to
one of the addresses that passed the check - never by handing the name to an HTTP library, which
would resolve it a second time. Without this, a hostile DNS server could answer with a public
address for the check and 192.168.1.1 for the connection (DNS rebinding). The name is still used
where it belongs: in the Host header, and for HTTPS as the TLS server name (SNI) and the name the
certificate must match, with full certificate verification against certifi's CA bundle.
(Fixed 2026-09-25; before that this was a documented residual risk.)

Config (fetch_url.json in your tool config folder - outside the repo), all optional:
    {
        "user_agent": "Mozilla/5.0 (compatible; amadeo-tool-server)",
        "max_bytes": 3000000,
        "max_chars": 20000,
        "allow_private_hosts": []
    }

Usage:
    fetch_url.py --describe
    echo '{"url": "https://example.com"}' | fetch_url.py --config fetch_url.json
"""
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import certifi
import urllib3
from requests.utils import get_encoding_from_headers

from amadeo_utils.ai.llm.tools.script_tools import ToolAnswer, ToolError, tool_script_main

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; amadeo-tool-server fetch_url)"
DEFAULT_MAX_BYTES = 3_000_000
DEFAULT_MAX_CHARS = 20_000
MAX_REDIRECTS = 5
MAX_ADDRESSES_TRIED = 3         # of a host's checked addresses, per hop - bounds the time spent on dead ones
HTTP_TIMEOUT = (5, 15)          # (connect, read) seconds, per request
REDIRECT_STATUSES = (301, 302, 303, 307, 308)
TEXT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/json", "text/xml",
              "application/xml", "text/markdown")

DEFINITION = {
    "name": "fetch_url",
    "description": "Fetches one public web page (http or https) and returns its readable text and title.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The full URL, starting http:// or https://"}},
        "required": ["url"],
    },
    "flags": {"outbound": True, "untrusted_output": True},
    "timeout_s": 30,
}


def check_url(url, config):
    """
    Refuses anything but a public http(s) URL - see "What it refuses" above - and returns the addresses it checked.

    This is the ONLY place the host name is resolved; the caller must connect to one of the returned addresses.

    Returns:
        tuple[str, int, list[str]]: host name, port, and the resolved addresses in resolver order (all of them
            checked, unless the host is in 'allow_private_hosts').

    Raises:
        ToolError: naming the reason, never echoing credentials.
    """
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise ToolError("only http:// and https:// URLs can be fetched")
    if parts.username or parts.password:
        raise ToolError("URLs containing a user name or password are not fetched")
    host = parts.hostname
    if not host:
        raise ToolError("the URL has no host name")
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, ValueError):
        raise ToolError(f"could not find the host {host}")
    addresses = list(dict.fromkeys(info[4][0] for info in infos))       # de-duplicated, resolver order kept
    if not addresses:
        raise ToolError(f"could not find the host {host}")
    if host.lower() not in {h.lower() for h in config.get("allow_private_hosts", [])}:
        for address in addresses:
            ip = ipaddress.ip_address(address.split("%")[0])
            if not ip.is_global or ip.is_multicast:
                raise ToolError(f"{host} is on a private or local network; only public web pages can be fetched")
    return host, port, addresses


def make_pool(scheme, address, port, host):
    """
    A one-connection pool pinned to an already-checked IP address.

    For HTTPS the certificate is verified against certifi's CA bundle, and must match 'host' - the name from the
    URL, which is also sent as the TLS server name (SNI) - not the IP address the connection actually goes to.
    """
    timeout = urllib3.Timeout(connect=HTTP_TIMEOUT[0], read=HTTP_TIMEOUT[1])
    if scheme == "https":
        return urllib3.HTTPSConnectionPool(address, port, timeout=timeout, retries=False, maxsize=1,
                                           cert_reqs="CERT_REQUIRED", ca_certs=certifi.where(),
                                           server_hostname=host, assert_hostname=host)
    return urllib3.HTTPConnectionPool(address, port, timeout=timeout, retries=False, maxsize=1)


def host_header(host, port, scheme):
    """The Host header for the URL's own name: brackets for an IPv6 literal, the port only when it is not the default."""
    name = f"[{host}]" if ":" in host else host
    return name if port == (443 if scheme == "https" else 80) else f"{name}:{port}"


def open_pinned(url, host, port, addresses, headers):
    """
    Sends the GET to the first of 'addresses' that accepts a connection (at most MAX_ADDRESSES_TRIED of them).

    Returns:
        tuple[urllib3.HTTPConnectionPool, urllib3.BaseHTTPResponse]: the pool (caller closes it) and the unread
            response.

    Raises:
        ToolError: if no address could be reached, or the request failed (TLS, timeout, protocol).
    """
    parts = urlsplit(url)
    target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
    headers = dict(headers, Host=host_header(host, port, parts.scheme))
    last_error = None
    for address in addresses[:MAX_ADDRESSES_TRIED]:
        pool = make_pool(parts.scheme, address, port, host)
        try:
            response = pool.urlopen("GET", target, headers=headers, redirect=False, retries=False,
                                    preload_content=False, assert_same_host=False)
            return pool, response
        except (urllib3.exceptions.NewConnectionError, urllib3.exceptions.ConnectTimeoutError) as e:
            pool.close()
            last_error = e                          # this address is unreachable; try the next checked one
        except urllib3.exceptions.HTTPError as e:
            pool.close()
            raise ToolError(f"the page could not be fetched ({type(e).__name__})")
    raise ToolError(f"the page could not be fetched ({type(last_error).__name__})")


def download(url, config):
    """
    GETs the URL, following redirects by hand so every hop passes check_url and is fetched from the address that
    was checked, and stops at max_bytes.

    Returns:
        tuple[str, str, bytes, bool, str]: final URL, content type, body, whether the body was cut short, encoding.
    """
    headers = {"User-Agent": config.get("user_agent") or DEFAULT_USER_AGENT,
               "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
               "Accept-Encoding": "gzip, deflate"}
    max_bytes = int(config.get("max_bytes", DEFAULT_MAX_BYTES))
    for _ in range(MAX_REDIRECTS + 1):
        host, port, addresses = check_url(url, config)
        pool, response = open_pinned(url, host, port, addresses, headers)
        try:
            if response.status in REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if not location:
                    raise ToolError(f"HTTP {response.status} redirect without a destination")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ToolError(f"the site answered HTTP {response.status}")
            content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type and content_type not in TEXT_TYPES:
                raise ToolError(f"the page is '{content_type}', not text, so it was not read")
            body, truncated = b"", False
            try:
                for chunk in response.stream(65536, decode_content=True):
                    body += chunk
                    if len(body) >= max_bytes:
                        body, truncated = body[:max_bytes], True
                        break
            except urllib3.exceptions.HTTPError as e:
                raise ToolError(f"the page could not be fetched ({type(e).__name__})")
            return url, content_type, body, truncated, get_encoding_from_headers(response.headers) or "utf-8"
        finally:
            response.close()
            pool.close()
    raise ToolError(f"more than {MAX_REDIRECTS} redirects")


def handle(arguments, config):
    """The tool: fetch, then reduce HTML to its readable text (headings and paragraphs kept, menus and scripts dropped)."""
    url = str(arguments.get("url", "")).strip()
    if not url:
        raise ToolError("a url is required")
    final_url, content_type, body, truncated, encoding = download(url, config)
    text = body.decode(encoding, errors="replace")
    title = None
    if content_type in ("text/html", "application/xhtml+xml", ""):
        import trafilatura   # deferred: only HTML needs it
        metadata = trafilatura.extract_metadata(text)
        title = metadata.title if metadata else None
        extracted = trafilatura.extract(text, include_formatting=True, include_tables=True, include_comments=False)
        if not extracted:
            raise ToolError("no readable text was found on that page")
        text = extracted
    max_chars = int(config.get("max_chars", DEFAULT_MAX_CHARS))
    if len(text) > max_chars:
        text, truncated = text[:max_chars], True
    # The history line describes the fetch and quotes nothing from the page - not even its title, which the page's
    # author wrote. Only the URL (already in the call's arguments or a redirect of it) and a size.
    summary = f"fetched {final_url} ({len(text):,} characters of text{', cut short' if truncated else ''})"
    return ToolAnswer({"url": final_url, "title": title, "text": text, "truncated": truncated}, summary)


if __name__ == "__main__":
    tool_script_main(DEFINITION, handle)
