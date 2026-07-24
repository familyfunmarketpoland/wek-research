from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_lab.data_guard import (
    DataGuardError,
    FinalAccessError,
    FINAL_CLAIM_FILENAME,
    ResearchAccessError,
    administrative_split,
    build_candidate_hash,
    load_holdout_once,
    load_research_dataset,
    load_research_path,
    manifest_fingerprint,
)


def _frame(start: str = "2025-01-01", periods: int = 10) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="MS", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": range(periods),
            "high": range(1, periods + 1),
            "low": range(periods),
            "close": range(1, periods + 1),
            "volume": [100.0 + offset for offset in range(periods)],
        },
        index=index,
        dtype=float,
    )


def _split_tmp_dataset(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _frame().to_parquet(data_dir / "btc_usdt_1d.parquet")
    plans = administrative_split(data_dir=data_dir)
    assert len(plans) == 1
    assert not (data_dir / "btc_usdt_1d.parquet").exists()
    return data_dir


def _split_two_tmp_datasets(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _frame().to_parquet(data_dir / "btc_usdt_1d.parquet")
    _frame(start="2024-01-01").to_parquet(data_dir / "eth_usdt_1d.parquet")
    plans = administrative_split(data_dir=data_dir)
    assert {plan.dataset for plan in plans} == {"btc_usdt_1d", "eth_usdt_1d"}
    assert not (data_dir / "btc_usdt_1d.parquet").exists()
    assert not (data_dir / "eth_usdt_1d.parquet").exists()
    return data_dir


def test_administrative_split_writes_manifest_and_removes_full_cache(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    manifest = json.loads((data_dir / "holdout_manifest.json").read_text(encoding="utf-8"))
    entry = manifest["datasets"]["btc_usdt_1d"]

    assert entry["cutoff"].startswith("2025-04-01")
    assert entry["source"]["rows"] == 10
    assert entry["research"]["rows"] == 4
    assert entry["holdout"]["rows"] == 6
    assert len(entry["research"]["sha256"]) == 64
    assert (data_dir / entry["research"]["path"]).exists()
    assert (data_dir / entry["holdout"]["path"]).exists()


def test_research_loader_accepts_only_manifest_research_file(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)

    loaded = load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)

    assert len(loaded) == 4
    assert loaded.index.max() == pd.Timestamp("2025-04-01 00:00:00+00:00")


def test_research_loader_rejects_holdout_paths_and_traversal(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    holdout_path = data_dir / "holdout" / "btc_usdt_1d.parquet"
    traversal = data_dir / "research" / ".." / "holdout" / "btc_usdt_1d.parquet"

    with pytest.raises(ResearchAccessError, match="outside data/research"):
        load_research_path(holdout_path, data_dir=data_dir)
    with pytest.raises(ResearchAccessError, match="outside data/research"):
        load_research_path(traversal, data_dir=data_dir)


def test_research_loader_rejects_symlink_and_hardlink(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    manifest_path = data_dir / "holdout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original = data_dir / "research" / "btc_usdt_1d.parquet"

    backup = data_dir / "research" / "backup.parquet"
    original.replace(backup)
    original.symlink_to(backup)
    with pytest.raises(DataGuardError, match="symlink"):
        load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)

    original.unlink()
    original.symlink_to(data_dir / "holdout" / "btc_usdt_1d.parquet")
    with pytest.raises(DataGuardError, match="outside data/research|symlink|identity mismatch"):
        load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)

    original.unlink()
    hardlink_target = data_dir / "hardlinked.parquet"
    backup.replace(data_dir / "holdout" / "btc_usdt_1d.parquet")
    (data_dir / "holdout" / "btc_usdt_1d.parquet").replace(hardlink_target)
    original.hardlink_to(hardlink_target)
    manifest["datasets"]["btc_usdt_1d"]["research"]["path"] = "research/btc_usdt_1d.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataGuardError, match="hardlinked|identity mismatch|hash mismatch"):
        load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)


def test_research_loader_rejects_rows_after_cutoff_and_hash_mismatch(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    research_path = data_dir / "research" / "btc_usdt_1d.parquet"
    holdout_rows = pd.read_parquet(data_dir / "holdout" / "btc_usdt_1d.parquet")
    holdout_rows.to_parquet(research_path)

    with pytest.raises(DataGuardError, match="identity mismatch|hash mismatch"):
        load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)

    manifest_path = data_dir / "holdout_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stat = research_path.stat()
    manifest["datasets"]["btc_usdt_1d"]["research"]["identity"] = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    manifest["datasets"]["btc_usdt_1d"]["research"]["sha256"] = build_candidate_hash(research_path)
    manifest["datasets"]["btc_usdt_1d"]["research"]["rows"] = len(holdout_rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ResearchAccessError, match="after cutoff"):
        load_research_dataset("BTC/USDT", "1d", data_dir=data_dir)


def test_final_loader_requires_candidate_hash_and_burns_once_before_read(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    active_manifest_fingerprint = manifest_fingerprint(data_dir / "holdout_manifest.json")
    candidate = {
        "dataset": "btc_usdt_1d",
        "params": {"length": 14},
        "manifest_fingerprint": active_manifest_fingerprint,
    }
    candidate_hash = build_candidate_hash(candidate)
    unbound_candidate = {"params": {"length": 14}}
    unbound_hash = build_candidate_hash(unbound_candidate)
    mismatched_candidate = {
        "dataset": "eth_usdt_1d",
        "params": {"length": 14},
        "manifest_fingerprint": active_manifest_fingerprint,
    }
    mismatched_hash = build_candidate_hash(mismatched_candidate)

    with pytest.raises(FinalAccessError, match="must be a lowercase"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash="bad",
            candidate=candidate,
        )
    with pytest.raises(FinalAccessError, match="does not match"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=candidate_hash,
            candidate={"different": True},
        )
    with pytest.raises(FinalAccessError, match="must include either dataset or symbol/timeframe"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=unbound_hash,
            candidate=unbound_candidate,
        )
    with pytest.raises(FinalAccessError, match="does not match requested holdout"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=mismatched_hash,
            candidate=mismatched_candidate,
        )

    loaded = load_holdout_once(
        "BTC/USDT",
        "1d",
        data_dir=data_dir,
        frozen_candidate_hash=candidate_hash,
        candidate=candidate,
    )

    assert len(loaded) == 6
    claim = data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME
    assert claim.exists()
    claim_payload = json.loads(claim.read_text(encoding="utf-8"))
    assert claim_payload["dataset"] == "btc_usdt_1d"
    assert claim_payload["frozen_candidate_hash"] == candidate_hash
    assert claim_payload["manifest_fingerprint"] == active_manifest_fingerprint
    with pytest.raises(FinalAccessError, match="already claimed"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=candidate_hash,
            candidate=candidate,
        )


def test_final_loader_global_claim_blocks_different_dataset_and_hash(tmp_path: Path) -> None:
    data_dir = _split_two_tmp_datasets(tmp_path)
    first_candidate = {
        "symbol": "BTC/USDT",
        "timeframe": "1d",
        "params": {"length": 14},
        "manifest_fingerprint": manifest_fingerprint(data_dir / "holdout_manifest.json"),
    }
    second_candidate = {
        "dataset": "eth_usdt_1d",
        "params": {"length": 20},
        "manifest_fingerprint": manifest_fingerprint(data_dir / "holdout_manifest.json"),
    }
    first_hash = build_candidate_hash(first_candidate)
    second_hash = build_candidate_hash(second_candidate)

    first = load_holdout_once(
        "BTC/USDT",
        "1d",
        data_dir=data_dir,
        frozen_candidate_hash=first_hash,
        candidate=first_candidate,
    )

    assert len(first) == 6
    with pytest.raises(FinalAccessError, match="already claimed"):
        load_holdout_once(
            "ETH/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=second_hash,
            candidate=second_candidate,
        )
    claims = list((data_dir / "holdout" / ".claims").glob("*.claimed"))
    assert [claim.name for claim in claims] == [FINAL_CLAIM_FILENAME]


def test_final_loader_rejects_non_mapping_candidate_before_claim(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    candidate = "unbound-candidate"

    with pytest.raises(FinalAccessError, match="frozen mapping"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=build_candidate_hash(candidate),
            candidate=candidate,  # type: ignore[arg-type]
        )

    assert not (data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME).exists()

    missing_manifest_candidate = {"dataset": "btc_usdt_1d"}
    with pytest.raises(FinalAccessError, match="manifest fingerprint"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=build_candidate_hash(missing_manifest_candidate),
            candidate=missing_manifest_candidate,
        )

    assert not (data_dir / "holdout" / ".claims" / FINAL_CLAIM_FILENAME).exists()


def test_final_loader_rejects_symlinked_claim_directory(tmp_path: Path) -> None:
    data_dir = _split_tmp_dataset(tmp_path)
    candidate = {
        "dataset": "btc_usdt_1d",
        "manifest_fingerprint": manifest_fingerprint(data_dir / "holdout_manifest.json"),
    }
    outside_claims = tmp_path / "outside-claims"
    outside_claims.mkdir()
    (data_dir / "holdout" / ".claims").symlink_to(outside_claims, target_is_directory=True)

    with pytest.raises(FinalAccessError, match="must not be a symlink"):
        load_holdout_once(
            "BTC/USDT",
            "1d",
            data_dir=data_dir,
            frozen_candidate_hash=build_candidate_hash(candidate),
            candidate=candidate,
        )

    assert not (outside_claims / FINAL_CLAIM_FILENAME).exists()
