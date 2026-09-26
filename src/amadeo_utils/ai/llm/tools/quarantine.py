"""
The quarantine store: where a delegate worker's answers live, out of the main model's reach.

Why it exists (CS-21, the dual-LLM pattern)
-------------------------------------------
A worker reads untrusted text - search results, fetched pages - and writes an answer derived
from it. If that answer entered the main model's context, injected text could steer the main
model, which may hold private data and may have tools that act on the machine. So the answer
goes straight to the user and is kept HERE; the main model's history gets only a marker naming
the record's id and the task it wrote itself. The main model can ask for a record to be shown
again ('recall') or used to seed a new worker ('delegate' with context), but no code path
returns a record's answer into the main model's messages.

Ids only ever increase and are never reused, so a stale marker can only miss - it can never
fetch some other answer. Records are evicted oldest-first past a count or an age.
"""
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


@dataclass
class DelegateRecord:
    """
    One worker answer.

    Attributes:
        task (str): What the main model asked the worker to do. Trusted - the main model wrote it.
        answer (str): The worker's final answer. UNTRUSTED: it is only ever shown to the user.
        created (str): ISO 8601, seconds.
        sources (list[str]): What the worker consulted (e.g. URLs), for "where did that come from?".
    """
    task: str
    answer: str
    created: str
    sources: List[str] = field(default_factory=list)


class QuarantineStore:
    """
    One session's worker answers, keyed by a monotonic integer id.

    Not thread-safe by itself: it lives in a session dictionary, and every access happens under
    that session's lock (StreamBase.get_session_and_lock), like everything else in a session.
    """

    FILENAME = "quarantine.json"

    def __init__(self, max_records: int = 50, max_age_days: float = 7.0):
        self.records: "OrderedDict[int, DelegateRecord]" = OrderedDict()
        self.next_id = 1
        self.max_records = max_records
        self.max_age = timedelta(days=max_age_days)
        # Small per-session state saved alongside the records, e.g. the main loop's taint in 'direct'
        # mode, which must survive a save and reload or a reloaded session would silently unlock.
        self.extra: Dict[str, Any] = {}

    def add(self, task: str, answer: str, sources: Optional[List[str]] = None, now: Optional[datetime] = None) -> int:
        """Stores an answer and returns its new id."""
        now = now or datetime.now()
        record_id = self.next_id
        self.next_id += 1
        self.records[record_id] = DelegateRecord(task, answer, now.isoformat(timespec="seconds"), list(sources or []))
        self.evict(now)
        return record_id

    def get(self, record_id: int, now: Optional[datetime] = None) -> Optional[DelegateRecord]:
        """The record, or None if it never existed or has been evicted."""
        self.evict(now)
        return self.records.get(record_id)

    def evict(self, now: Optional[datetime] = None):
        """Drops records past the count limit, then any older than the age limit - oldest first either way."""
        now = now or datetime.now()
        while len(self.records) > self.max_records:
            self.records.popitem(last=False)
        while self.records:
            oldest = next(iter(self.records.values()))
            if now - datetime.fromisoformat(oldest.created) <= self.max_age:
                break
            self.records.popitem(last=False)

    # ---------------------------------------------------------------------------------- persistence

    def to_dict(self) -> Dict[str, Any]:
        """Serialisable form. 'next_id' is saved too: without it a reload would reuse old ids."""
        return {"next_delegate_id": self.next_id,
                "records": {str(k): asdict(v) for k, v in self.records.items()},
                "extra": self.extra}

    @classmethod
    def from_dict(cls, data: Dict[str, Any], max_records: int = 50, max_age_days: float = 7.0) -> "QuarantineStore":
        store = cls(max_records, max_age_days)
        store.next_id = int(data.get("next_delegate_id", 1))
        store.extra = dict(data.get("extra") or {})
        for key in sorted(data.get("records", {}), key=int):
            store.records[int(key)] = DelegateRecord(**data["records"][key])
        store.evict()
        return store

    def save(self, directory: str, encryption=None):
        """
        Writes the store to 'directory'. Called only when the user has asked for the conversation to be
        saved, and BEFORE the chat history and vector database, so a crash part-way through can orphan
        a record but never leave a marker pointing at one that was not written.

        Args:
            directory: The session's conversation directory.
            encryption: The session VectorDB's AmadeoEncryption, when a passphrase is set; the file is
                then written encrypted with a '.enc' suffix, like the history beside it.
        """
        os.makedirs(directory, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if encryption is not None:
            encryption.encrypt_and_save(text, os.path.join(directory, self.FILENAME + ".enc"))
        else:
            with open(os.path.join(directory, self.FILENAME), "w", encoding="utf-8") as fh:
                fh.write(text)

    @classmethod
    def load(cls, directory: str, encryption=None, max_records: int = 50, max_age_days: float = 7.0) -> "QuarantineStore":
        """Reads a saved store, or returns an empty one if there is none."""
        if encryption is not None:
            path = os.path.join(directory, cls.FILENAME + ".enc")
            if not os.path.exists(path):
                return cls(max_records, max_age_days)
            text = encryption.load_and_decrypt(path, return_type="str")
        else:
            path = os.path.join(directory, cls.FILENAME)
            if not os.path.exists(path):
                return cls(max_records, max_age_days)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        return cls.from_dict(json.loads(text), max_records, max_age_days)
