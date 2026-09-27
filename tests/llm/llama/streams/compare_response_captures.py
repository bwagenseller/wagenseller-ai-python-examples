#!/usr/bin/env python3
"""CS-21: compare a fresh golden_response capture with an existing baseline.

Reports, per family: whether every key already in the baseline is identical in the fresh
capture (it must be - extending the harness may only ADD scenarios), which scenarios were
added, and for each added one whether the session's history still fits and how many messages
reached the model. Used when extending CS-20's golden_response.py with the overflow scenarios.

Usage:  python compare_response_captures.py BASELINE.json FRESH.json
"""
import json
import sys

old, new = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for family in ("RolePlayStream", "KnowledgeBaseStream"):
    same = all(new[family].get(k) == v for k, v in old[family].items())
    bad += 0 if same else 1
    print(f"{family}: {'old keys identical' if same else 'OLD KEYS CHANGED'}")
    for key in sorted(set(new[family]) - set(old[family])):
        record = new[family][key]
        calls = record.get("generator_calls") or []
        fits = (record.get("session_after") or {}).get("full_history_fits")
        messages = len(calls[0]["messages"]) if calls else 0
        print(f"    + {key:26} full_history_fits={fits}  messages_to_model={messages}  "
              f"reached_model={bool(calls)}  raised={'raised' in record}")
sys.exit(bad)
