from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


GENESIS_HEAD = "0" * 64


class IntegrityError(RuntimeError):
    """Raised when a forward-test artifact fails an integrity check."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_head(path: Path) -> str:
    if not path.exists():
        return GENESIS_HEAD
    head = path.read_text(encoding="utf-8").strip()
    if not _is_sha256(head):
        raise IntegrityError(f"invalid ledger head in {path}")
    return head


def ledger_entry_hash(prev_hash: str, entry_without_hash: dict[str, Any]) -> str:
    if not _is_sha256(prev_hash):
        raise IntegrityError("invalid previous ledger hash")
    return sha256_bytes(prev_hash.encode("ascii") + b"\n" + canonical_json_bytes(entry_without_hash))


def build_ledger_entry(sequence: int, prev_hash: str, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    entry = {
        "schema_version": 1,
        "sequence": int(sequence),
        "prev_hash": prev_hash,
        "event": event,
        "payload": payload,
    }
    entry["entry_hash"] = ledger_entry_hash(prev_hash, entry)
    return entry


def verify_ledger(path: Path, head_path: Path) -> tuple[str, int, list[dict[str, Any]]]:
    expected_prev = GENESIS_HEAD
    entries: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    raise IntegrityError(f"blank ledger line at {line_number}")
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise IntegrityError(f"invalid ledger JSON at line {line_number}") from exc
                _verify_entry(entry, expected_prev, len(entries), line_number)
                expected_prev = str(entry["entry_hash"])
                entries.append(entry)
    recorded_head = read_head(head_path)
    if recorded_head != expected_prev:
        raise IntegrityError("ledger head does not match ledger contents")
    return expected_prev, len(entries), entries


def append_ledger_entries(path: Path, entries: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False))
            handle.write("\n")


def write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")


def _verify_entry(entry: Any, expected_prev: str, expected_sequence: int, line_number: int) -> None:
    if not isinstance(entry, dict):
        raise IntegrityError(f"ledger line {line_number} is not an object")
    required = {"schema_version", "sequence", "prev_hash", "event", "payload", "entry_hash"}
    if set(entry) != required:
        raise IntegrityError(f"ledger line {line_number} has unexpected schema")
    if entry["schema_version"] != 1:
        raise IntegrityError(f"ledger line {line_number} has unsupported schema_version")
    if entry["sequence"] != expected_sequence:
        raise IntegrityError(f"ledger sequence mismatch at line {line_number}")
    if entry["prev_hash"] != expected_prev:
        raise IntegrityError(f"ledger previous hash mismatch at line {line_number}")
    entry_hash = entry["entry_hash"]
    if not _is_sha256(entry_hash):
        raise IntegrityError(f"ledger line {line_number} has invalid entry_hash")
    without_hash = dict(entry)
    without_hash.pop("entry_hash")
    if ledger_entry_hash(expected_prev, without_hash) != entry_hash:
        raise IntegrityError(f"ledger line {line_number} hash mismatch")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
