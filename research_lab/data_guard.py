"""Holdout isolation and guarded parquet access.

This module separates administrative data splitting from research/final-access
loaders. The administrative split may read full-history parquet caches and
write isolated research/holdout files. Ordinary research loaders only accept
files recorded under ``data/research`` and reject holdout paths, traversal,
links, stale file identities, hash mismatches, and timestamps after cutoff.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESEARCH_DIRNAME = "research"
HOLDOUT_DIRNAME = "holdout"
MANIFEST_FILENAME = "holdout_manifest.json"
CLAIMS_DIRNAME = ".claims"
FINAL_CLAIM_FILENAME = "final_holdout_access.claimed"
PARQUET_SUFFIX = ".parquet"
DATASET_RE = re.compile(r"^[a-z0-9]+_[a-z0-9]+_[0-9]+[a-z]+$")


class DataGuardError(RuntimeError):
    """Base class for guarded-data failures."""


class ResearchAccessError(DataGuardError):
    """Raised when research access would violate holdout isolation."""


class FinalAccessError(DataGuardError):
    """Raised when final holdout access is not explicitly authorized."""


@dataclass(frozen=True)
class SplitPlan:
    """Summary for one administratively split parquet cache."""

    dataset: str
    cutoff: str
    source_path: str
    research_path: str
    holdout_path: str
    source_rows: int
    research_rows: int
    holdout_rows: int
    source_sha256: str
    research_sha256: str
    holdout_sha256: str


@dataclass(frozen=True)
class HoldoutManifest:
    """Loaded holdout manifest with its canonical filesystem location."""

    path: Path
    data_dir: Path
    entries: Mapping[str, Mapping[str, Any]]


def build_candidate_hash(candidate: bytes | str | Path | Mapping[str, Any]) -> str:
    """Return a stable SHA256 hash for a frozen candidate artifact.

    ``Path`` values hash file bytes. Mapping values are JSON-canonicalized.
    Strings and bytes hash their exact UTF-8/byte representation.
    """

    if isinstance(candidate, Path):
        return _sha256_file(candidate)
    if isinstance(candidate, Mapping):
        return _sha256_json_value(candidate)
    elif isinstance(candidate, bytes):
        payload = candidate
    elif isinstance(candidate, str):
        payload = candidate.encode("utf-8")
    else:
        raise TypeError("candidate must be bytes, str, Path, or Mapping")
    return hashlib.sha256(payload).hexdigest()


def manifest_fingerprint(manifest: HoldoutManifest | Mapping[str, Any] | Path) -> str:
    """Return a SHA256 fingerprint for a split manifest.

    Manifest JSON is canonicalized before hashing, so pretty-printing does not
    change the fingerprint. Loaded ``HoldoutManifest`` values hash their
    manifest file's parsed JSON.
    """

    if isinstance(manifest, HoldoutManifest):
        return manifest_fingerprint(manifest.path)
    if isinstance(manifest, Path):
        try:
            return _sha256_json_value(json.loads(manifest.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise DataGuardError(f"invalid manifest JSON at {manifest}: {exc.msg}") from exc
    if isinstance(manifest, Mapping):
        return _sha256_json_value(manifest)
    raise TypeError("manifest must be a HoldoutManifest, Mapping, or Path")


def administrative_split(
    *,
    data_dir: str | Path = DATA_DIR,
    cutoff_months: int = 6,
    manifest_name: str = MANIFEST_FILENAME,
) -> list[SplitPlan]:
    """Split full caches into research and holdout parquet files.

    This is the only API in this module intended to read unsplit full-history
    parquet caches. It skips existing ``research`` and ``holdout`` directories,
    writes outputs atomically, writes a manifest atomically, and removes the
    source parquet only after both split files and the manifest are durable.
    """

    if cutoff_months <= 0:
        raise ValueError("cutoff_months must be positive")

    root = _canonical_existing_dir(Path(data_dir), label="data_dir")
    research_dir = root / RESEARCH_DIRNAME
    holdout_dir = root / HOLDOUT_DIRNAME
    research_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(
        path
        for path in root.glob(f"*{PARQUET_SUFFIX}")
        if path.is_file() and not path.is_symlink()
    )
    plans: list[SplitPlan] = []
    entries: dict[str, dict[str, Any]] = {}
    manifest_path = root / manifest_name

    for source_path in source_paths:
        dataset = _dataset_from_filename(source_path.name)
        _reject_linked_file(source_path, label="source")
        source_stat = source_path.stat()
        source_sha = _sha256_file(source_path)
        full = _normalize_frame(pd.read_parquet(source_path), source_path)
        if full.empty:
            raise DataGuardError(f"cannot split empty dataset {source_path}")

        cutoff = _cutoff_for_frame(full, months=cutoff_months)
        research = full.loc[full.index <= cutoff]
        holdout = full.loc[full.index > cutoff]
        if research.empty:
            raise DataGuardError(f"split would leave empty research set for {dataset}")
        if holdout.empty:
            raise DataGuardError(f"split would leave empty holdout set for {dataset}")

        research_path = research_dir / source_path.name
        holdout_path = holdout_dir / source_path.name
        _atomic_write_parquet(research_path, research)
        _atomic_write_parquet(holdout_path, holdout)
        _reject_linked_file(research_path, label="research")
        _reject_linked_file(holdout_path, label="holdout")

        research_sha = _sha256_file(research_path)
        holdout_sha = _sha256_file(holdout_path)
        research_stat = research_path.stat()
        holdout_stat = holdout_path.stat()

        entry = {
            "dataset": dataset,
            "cutoff": cutoff.isoformat(),
            "cutoff_months": cutoff_months,
            "source": {
                "path": source_path.name,
                "rows": int(len(full)),
                "sha256": source_sha,
                "identity": _identity(source_stat),
            },
            "research": {
                "path": f"{RESEARCH_DIRNAME}/{source_path.name}",
                "rows": int(len(research)),
                "sha256": research_sha,
                "identity": _identity(research_stat),
            },
            "holdout": {
                "path": f"{HOLDOUT_DIRNAME}/{source_path.name}",
                "rows": int(len(holdout)),
                "sha256": holdout_sha,
                "identity": _identity(holdout_stat),
            },
        }
        entries[dataset] = entry
        plans.append(
            SplitPlan(
                dataset=dataset,
                cutoff=cutoff.isoformat(),
                source_path=str(source_path),
                research_path=str(research_path),
                holdout_path=str(holdout_path),
                source_rows=int(len(full)),
                research_rows=int(len(research)),
                holdout_rows=int(len(holdout)),
                source_sha256=source_sha,
                research_sha256=research_sha,
                holdout_sha256=holdout_sha,
            )
        )

    if not plans:
        raise DataGuardError(f"no full-history parquet caches found in {root}")

    manifest = {
        "schema_version": 1,
        "step": "administrative_split",
        "data_dir": str(root),
        "research_dir": RESEARCH_DIRNAME,
        "holdout_dir": HOLDOUT_DIRNAME,
        "datasets": entries,
    }
    _atomic_write_json(manifest_path, manifest)

    for plan in plans:
        source = Path(plan.source_path)
        if source.exists():
            source.unlink()

    return plans


def load_research_dataset(
    symbol: str,
    timeframe: str,
    *,
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load one research dataset only after validating holdout isolation."""

    dataset = _dataset_from_symbol_timeframe(symbol, timeframe)
    manifest = load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    entry = _entry_for_dataset(manifest, dataset)
    path = _validated_manifest_path(manifest, entry, "research")
    frame = _read_guarded_parquet(path, entry=entry, section="research")
    cutoff = _parse_utc_timestamp(entry["cutoff"], label="cutoff")
    if not frame.empty and frame.index.max() > cutoff:
        raise ResearchAccessError(
            f"research dataset {dataset} contains rows after cutoff {cutoff.isoformat()}"
        )
    expected_rows = int(entry["research"]["rows"])
    if len(frame) != expected_rows:
        raise ResearchAccessError(
            f"research dataset {dataset} row count mismatch: expected {expected_rows}, got {len(frame)}"
        )
    return frame


def load_research_path(
    path: str | Path,
    *,
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load a research parquet by path, rejecting traversal and holdout paths."""

    manifest = load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    candidate = _canonical_guarded_file(Path(path), root=manifest.data_dir, section=RESEARCH_DIRNAME)
    dataset = _dataset_from_filename(candidate.name)
    entry = _entry_for_dataset(manifest, dataset)
    expected = _validated_manifest_path(manifest, entry, "research")
    if candidate != expected:
        raise ResearchAccessError(f"path {candidate} is not the manifest research path for {dataset}")
    frame = _read_guarded_parquet(candidate, entry=entry, section="research")
    cutoff = _parse_utc_timestamp(entry["cutoff"], label="cutoff")
    if not frame.empty and frame.index.max() > cutoff:
        raise ResearchAccessError(
            f"research dataset {dataset} contains rows after cutoff {cutoff.isoformat()}"
        )
    return frame


def load_holdout_once(
    symbol: str,
    timeframe: str,
    *,
    frozen_candidate_hash: str,
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path | None = None,
    candidate: bytes | str | Path | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Burn a one-time final-access claim, then load exactly one holdout dataset.

    ``frozen_candidate_hash`` must be the SHA256 of the already-frozen candidate
    artifact. If ``candidate`` is supplied, it is hashed and must match the
    provided hash before the claim is burned.
    """

    _validate_sha256(frozen_candidate_hash, label="frozen_candidate_hash")
    if candidate is not None:
        actual_candidate_hash = build_candidate_hash(candidate)
        if actual_candidate_hash != frozen_candidate_hash:
            raise FinalAccessError("frozen_candidate_hash does not match candidate")

    dataset = _dataset_from_symbol_timeframe(symbol, timeframe)
    if isinstance(candidate, Mapping):
        _validate_candidate_dataset_binding(candidate, dataset=dataset, symbol=symbol, timeframe=timeframe)
    manifest = load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    fingerprint = manifest_fingerprint(manifest)
    if isinstance(candidate, Mapping):
        _validate_candidate_manifest_binding(candidate, fingerprint=fingerprint)
    entry = _entry_for_dataset(manifest, dataset)
    claim_path = _burn_final_claim(
        manifest.data_dir,
        dataset=dataset,
        frozen_candidate_hash=frozen_candidate_hash,
        manifest_fingerprint=fingerprint,
    )
    try:
        path = _validated_manifest_path(manifest, entry, "holdout")
        frame = _read_guarded_parquet(path, entry=entry, section="holdout")
    except Exception as exc:
        raise FinalAccessError(
            f"final claim was burned at {claim_path}, but holdout load failed: {exc}"
        ) from exc

    cutoff = _parse_utc_timestamp(entry["cutoff"], label="cutoff")
    if not frame.empty and frame.index.min() <= cutoff:
        raise FinalAccessError(
            f"holdout dataset {dataset} contains rows at or before cutoff {cutoff.isoformat()}"
        )
    expected_rows = int(entry["holdout"]["rows"])
    if len(frame) != expected_rows:
        raise FinalAccessError(
            f"holdout dataset {dataset} row count mismatch: expected {expected_rows}, got {len(frame)}"
        )
    return frame


def load_manifest(
    *,
    data_dir: str | Path = DATA_DIR,
    manifest_path: str | Path | None = None,
) -> HoldoutManifest:
    root = _canonical_existing_dir(Path(data_dir), label="data_dir")
    path = Path(manifest_path) if manifest_path is not None else root / MANIFEST_FILENAME
    manifest_file = _canonical_child_file(path, root=root, label="manifest")
    _reject_linked_file(manifest_file, label="manifest")
    try:
        raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DataGuardError(f"manifest not found at {manifest_file}") from exc
    except json.JSONDecodeError as exc:
        raise DataGuardError(f"invalid manifest JSON at {manifest_file}: {exc.msg}") from exc

    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DataGuardError("manifest must be a schema_version=1 JSON object")
    entries = raw.get("datasets")
    if not isinstance(entries, dict) or not entries:
        raise DataGuardError("manifest must contain non-empty datasets")
    return HoldoutManifest(path=manifest_file, data_dir=root, entries=entries)


def _read_guarded_parquet(
    path: Path,
    *,
    entry: Mapping[str, Any],
    section: str,
) -> pd.DataFrame:
    _reject_linked_file(path, label=section)
    section_data = entry[section]
    expected_identity = section_data.get("identity")
    if not isinstance(expected_identity, Mapping):
        raise DataGuardError(f"manifest {section} identity is missing")
    if _identity(path.stat()) != dict(expected_identity):
        raise DataGuardError(f"{section} file identity mismatch for {path}")
    expected_sha = section_data.get("sha256")
    if not isinstance(expected_sha, str):
        raise DataGuardError(f"manifest {section} sha256 is missing")
    if _sha256_file(path) != expected_sha:
        raise DataGuardError(f"{section} file hash mismatch for {path}")
    return _normalize_frame(pd.read_parquet(path), path)


def _validated_manifest_path(manifest: HoldoutManifest, entry: Mapping[str, Any], section: str) -> Path:
    section_data = entry.get(section)
    if not isinstance(section_data, Mapping):
        raise DataGuardError(f"manifest entry missing {section} section")
    rel = section_data.get("path")
    if not isinstance(rel, str):
        raise DataGuardError(f"manifest {section} path must be a string")
    section_dir = RESEARCH_DIRNAME if section == "research" else HOLDOUT_DIRNAME
    try:
        candidate = manifest.data_dir / rel
        return _canonical_guarded_file(candidate, root=manifest.data_dir, section=section_dir)
    except ResearchAccessError as exc:
        raise DataGuardError(f"invalid manifest {section} path {rel!r}: {exc}") from exc


def _canonical_guarded_file(path: Path, *, root: Path, section: str) -> Path:
    if section not in {RESEARCH_DIRNAME, HOLDOUT_DIRNAME}:
        raise ValueError("section must be research or holdout")
    root = root.resolve(strict=True)
    if path.is_symlink():
        raise ResearchAccessError(f"guarded path must not be a symlink: {path}")
    try:
        canonical = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ResearchAccessError(f"guarded parquet not found: {path}") from exc
    allowed_root = (root / section).resolve(strict=True)
    if not _is_relative_to(canonical, allowed_root):
        raise ResearchAccessError(f"path {canonical} is outside data/{section}")
    if section == RESEARCH_DIRNAME and HOLDOUT_DIRNAME in canonical.relative_to(root).parts:
        raise ResearchAccessError("research loader cannot access holdout paths")
    if canonical.suffix != PARQUET_SUFFIX:
        raise ResearchAccessError(f"guarded path must be a parquet file: {canonical}")
    return canonical


def _burn_final_claim(
    data_dir: Path,
    *,
    dataset: str,
    frozen_candidate_hash: str,
    manifest_fingerprint: str,
) -> Path:
    claims_dir = data_dir / HOLDOUT_DIRNAME / CLAIMS_DIRNAME
    claims_dir.mkdir(parents=True, exist_ok=True)
    claims_dir = claims_dir.resolve(strict=True)
    claim_path = claims_dir / FINAL_CLAIM_FILENAME
    payload = json.dumps(
        {
            "dataset": dataset,
            "frozen_candidate_hash": frozen_candidate_hash,
            "manifest_fingerprint": manifest_fingerprint,
            "claim": "final_holdout_access",
        },
        sort_keys=True,
        indent=2,
    )
    try:
        fd = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise FinalAccessError(
            "final holdout access already claimed for this study"
        ) from exc
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(claim_path, 0o400)
    return claim_path


def _normalize_frame(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise DataGuardError(f"parquet at {path} must use a DatetimeIndex")
    normalized = frame.copy()
    if normalized.index.tz is None:
        normalized.index = normalized.index.tz_localize("UTC")
    else:
        normalized.index = normalized.index.tz_convert("UTC")
    if normalized.index.has_duplicates:
        raise DataGuardError(f"parquet at {path} has duplicate timestamps")
    if not normalized.index.is_monotonic_increasing:
        normalized = normalized.sort_index()
    normalized.index.name = "timestamp"
    return normalized


def _cutoff_for_frame(frame: pd.DataFrame, *, months: int) -> pd.Timestamp:
    latest = frame.index.max()
    return latest - pd.DateOffset(months=months)


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        frame.to_parquet(tmp_path)
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
        encoding="utf-8",
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(tmp_path, path)
        _fsync_dir(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _canonical_existing_dir(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DataGuardError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise DataGuardError(f"{label} must be a directory: {resolved}")
    return resolved


def _canonical_child_file(path: Path, *, root: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DataGuardError(f"{label} does not exist: {path}") from exc
    if not _is_relative_to(resolved, root):
        raise DataGuardError(f"{label} must be inside {root}: {resolved}")
    if not resolved.is_file():
        raise DataGuardError(f"{label} must be a file: {resolved}")
    return resolved


def _reject_linked_file(path: Path, *, label: str) -> None:
    if path.is_symlink():
        raise DataGuardError(f"{label} path must not be a symlink: {path}")
    stat_result = path.stat()
    if stat_result.st_nlink != 1:
        raise DataGuardError(f"{label} path must not be hardlinked: {path}")


def _identity(stat_result: os.stat_result) -> dict[str, int]:
    return {
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "size": int(stat_result.st_size),
        "mtime_ns": int(stat_result.st_mtime_ns),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json_value(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _dataset_from_symbol_timeframe(symbol: str, timeframe: str) -> str:
    if "/" not in symbol:
        raise ValueError(f"symbol must be in BASE/QUOTE form, got {symbol!r}")
    base, quote = symbol.split("/", 1)
    return _validate_dataset(f"{base.lower()}_{quote.lower()}_{timeframe.lower()}")


def _dataset_from_filename(filename: str) -> str:
    if not filename.endswith(PARQUET_SUFFIX):
        raise ValueError(f"dataset file must end with {PARQUET_SUFFIX}: {filename}")
    return _validate_dataset(filename[: -len(PARQUET_SUFFIX)])


def _validate_dataset(dataset: str) -> str:
    if not DATASET_RE.fullmatch(dataset):
        raise ValueError(f"invalid dataset id: {dataset!r}")
    return dataset


def _entry_for_dataset(manifest: HoldoutManifest, dataset: str) -> Mapping[str, Any]:
    entry = manifest.entries.get(dataset)
    if not isinstance(entry, Mapping):
        raise DataGuardError(f"dataset {dataset} is not present in holdout manifest")
    return entry


def _parse_utc_timestamp(value: Any, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if pd.isna(timestamp):
        raise DataGuardError(f"{label} is not a valid timestamp")
    return timestamp


def _validate_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise FinalAccessError(f"{label} must be a lowercase 64-character SHA256 hex digest")


def _validate_candidate_dataset_binding(
    candidate: Mapping[str, Any],
    *,
    dataset: str,
    symbol: str,
    timeframe: str,
) -> None:
    candidate_dataset = candidate.get("dataset")
    if candidate_dataset is not None:
        if _validate_dataset(str(candidate_dataset).lower()) != dataset:
            raise FinalAccessError(
                f"candidate dataset {candidate_dataset!r} does not match requested holdout {dataset!r}"
            )
        return

    candidate_symbol = candidate.get("symbol")
    candidate_timeframe = candidate.get("timeframe")
    if candidate_symbol is None or candidate_timeframe is None:
        raise FinalAccessError(
            "mapping candidate must include either dataset or symbol/timeframe for holdout binding"
        )
    candidate_bound_dataset = _dataset_from_symbol_timeframe(str(candidate_symbol), str(candidate_timeframe))
    if candidate_bound_dataset != dataset:
        raise FinalAccessError(
            f"candidate symbol/timeframe {candidate_symbol!r} {candidate_timeframe!r} "
            f"does not match requested holdout {symbol!r} {timeframe!r}"
        )


def _validate_candidate_manifest_binding(candidate: Mapping[str, Any], *, fingerprint: str) -> None:
    candidate_fingerprint = candidate.get("manifest_fingerprint")
    if candidate_fingerprint is None:
        candidate_fingerprint = candidate.get("manifest_sha256")
    if candidate_fingerprint is not None and candidate_fingerprint != fingerprint:
        raise FinalAccessError("candidate manifest fingerprint does not match active holdout manifest")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_split_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Administratively split full caches into research/holdout.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--cutoff-months", type=int, default=6)
    parser.add_argument("--manifest-name", default=MANIFEST_FILENAME)
    return parser.parse_args(argv)


def split_main(argv: Iterable[str] | None = None) -> int:
    args = _parse_split_args(argv)
    plans = administrative_split(
        data_dir=args.data_dir,
        cutoff_months=args.cutoff_months,
        manifest_name=args.manifest_name,
    )
    for plan in plans:
        print(
            f"{plan.dataset}: research_rows={plan.research_rows} "
            f"holdout_rows={plan.holdout_rows} cutoff={plan.cutoff}"
        )
    return 0
