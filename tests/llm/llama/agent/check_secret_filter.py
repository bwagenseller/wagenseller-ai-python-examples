#!/usr/bin/env python3
"""CS-21: the outbound credential backstop (registry.find_secret) - what it catches and what it must not.

The owner's rule: private data may go into a web search, passwords and keys may not. The filter
matches credential SHAPES, so the bare words "password" or "API key" in an ordinary question
must pass. Runs anywhere.
"""
import os
import sys

# testlib finds the library under test and this machine's paths (tests/README.md).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from testlib import settings  # noqa: E402
settings.use_src()
from amadeo_utils.ai.llm.tools.registry import find_secret   # noqa: E402

MUST_CATCH = [
    "password: hunter22", "my api_key=abcd1234", "PASSWD=letmein99", "secret key: 8f3k2j", "access_token=abc123xyz",
    "Authorization: Bearer abcdefghijklmnop1234", "-----BEGIN RSA PRIVATE KEY-----", "-----BEGIN OPENSSH PRIVATE KEY-----",
    "AKIAABCDEFGHIJKLMNOP", "ghp_" + "a" * 36, "github_pat_" + "b" * 30, "sk-ant-api03-abcdefghijklmnopqrstuv",
    "https://bob:pa55@example.com/x", "xoxb-1234567890-abc", "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
]
MUST_PASS = [
    "how do I reset my password", "what is an API key", "best password managers 2026", "the secret garden book",
    "token ring networks", "https://example.com/path?q=1", "Algebra 1 homework help for Q1", "passphrase ideas",
    "email bob@example.com about the game",
]

failures = [f"missed: {s!r}" for s in MUST_CATCH if not find_secret({"q": s})]
failures += [f"false positive: {s!r}" for s in MUST_PASS if find_secret({"q": s})]
if not find_secret({"outer": [{"inner": "password=xyzzy1"}]}):
    failures.append("missed a secret nested inside a list of dicts")
for f in failures:
    print("  FAIL ", f)
print(f"{len(MUST_CATCH)} must-catch, {len(MUST_PASS)} must-pass: " + ("ALL CHECKS PASS" if not failures else f"{len(failures)} FAILED"))
sys.exit(len(failures))
