"""Consistent live WAL snapshots, failures and isolated recovery; no network/orders."""

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from unittest.mock import Mock

import pytest

from crypto_arbitrage_bot import snapshot_backup as module
from crypto_arbitrage_bot.storage.opportunity_records import exact_json


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr("socket.create_connection", Mock(side_effect=AssertionError("snapshot is local only")))


@pytest.fixture
def project(tmp_path):
    path = tmp_path / "data/research/test.sqlite"
    path.parent.mkdir(parents=True)
    with closing(sqlite3.connect(path)) as db:
        db.executescript("""PRAGMA user_version=1;
            CREATE TABLE opportunities(id TEXT PRIMARY KEY, amount TEXT, missing TEXT, payload BLOB);
            CREATE TABLE opportunity_sources(parent TEXT REFERENCES opportunities(id));
            CREATE TABLE opportunity_scenarios(parent TEXT REFERENCES opportunities(id));""")
        db.execute("INSERT INTO opportunities VALUES (?,?,?,?)", ("original", "0.123456789012345678901234567890",
                                                                  None, b"\x00\xff"))
        db.execute("INSERT INTO opportunity_sources VALUES ('original')")
        db.commit()
    return tmp_path, path, tmp_path / "backups/snapshot"


def create(project, **kwargs):
    root, path, out = project
    result = module.create(root, (path,), out, min_free_bytes=0, **kwargs)
    return result, module.verify(out, result["manifest_sha256"])


def test_snapshot_verify_and_isolated_rehearsal_preserve_exact_values_and_original(project):
    root, path, out = project
    before = path.read_bytes()
    result, manifest = create(project)
    restored = root / "backups/rehearsal"
    proof = module.rehearse(root, out, result["manifest_sha256"], restored, min_free_bytes=0)
    assert proof["status"] == "ISOLATED_REHEARSAL_VERIFIED" and proof["live_recovery_authorized"] is False
    assert path.read_bytes() == before
    with closing(sqlite3.connect(restored / "data/research/test.sqlite")) as db:
        assert db.execute("SELECT * FROM opportunities").fetchall() == [(
            "original", "0.123456789012345678901234567890", None, b"\x00\xff")]
    entry = manifest["databases"][0]
    assert entry["audit"]["row_counts"]["opportunities"] == 1
    assert entry["audit"]["user_version"] == 1 and manifest["global_atomic_snapshot"] is False
    assert {p.name for p in out.iterdir()} == {"manifest.json", "database-00.sqlite"}


def test_committed_wal_data_and_concurrent_write_have_one_pinned_snapshot(project, monkeypatch):
    root, path, _ = project
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("INSERT INTO opportunities VALUES ('in-wal','2',NULL,NULL)")
    connection.commit()
    assert Path(str(path)+"-wal").is_file()
    original_connect = sqlite3.connect
    changes = []
    class LiveSource(sqlite3.Connection):
        def backup(self, target, **kwargs):
            progress = kwargs["progress"]
            def observe(*args):
                if not changes:
                    connection.execute("INSERT INTO opportunities VALUES ('later','3',NULL,NULL)")
                    connection.commit()
                    changes.append(True)
                progress(*args)
            kwargs["progress"] = observe
            return super().backup(target, **kwargs)
    def connect(database, **kwargs):
        if str(database) == path.as_uri()+"?mode=ro":
            kwargs["factory"] = LiveSource
        return original_connect(database, **kwargs)
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    try:
        result, manifest = create(project)
        assert changes and manifest["databases"][0]["audit"]["row_counts"]["opportunities"] == 2
        with closing(original_connect(Path(result["output_directory"]) / "database-00.sqlite")) as db:
            assert {r[0] for r in db.execute("SELECT id FROM opportunities")} == {"original", "in-wal"}
        assert connection.execute("SELECT count(*) FROM opportunities").fetchone()[0] == 3
    finally:
        connection.close()


def test_execution_attempts_and_states_are_not_reset(project):
    root, _, out = project
    path = root / "data/testnet/execution.sqlite"
    path.parent.mkdir()
    with closing(sqlite3.connect(path)) as db:
        db.executescript("CREATE TABLE testnet_pair_attempts(id TEXT,state TEXT); CREATE TABLE testnet_order_intents(id TEXT);")
        db.executemany("INSERT INTO testnet_pair_attempts VALUES (?,?)", [(str(i), "CLOSED") for i in range(20)])
        db.commit()
    result = module.create(root, (path,), out, min_free_bytes=0)
    manifest = module.verify(out, result["manifest_sha256"])
    assert manifest["databases"][0]["audit"]["row_counts"]["testnet_pair_attempts"] == 20
    assert manifest["databases"][0]["audit"]["kind"] == "TESTNET_EXECUTION_ARCHIVE"


@pytest.mark.parametrize("target", ["data/research/new", "src/new", "docs/new", ".", "outside"])
def test_target_must_be_new_and_isolated(project, target):
    root, path, _ = project
    output = root.parent / "outside" if target == "outside" else root / target
    with pytest.raises(ValueError, match="ISOLATED_TARGET"):
        module.create(root, (path,), output)
    assert path.is_file()


@pytest.mark.parametrize("name", ["data/research/../test.sqlite", "C:/escape.sqlite", "/escape.sqlite",
    "data/testnet/credentials.sqlite", "data/testnet/secrets.sqlite", "data/testnet/credentials.dpapi",
    "data\\research\\test.sqlite", "data/research//test.sqlite", "data/other/test.sqlite"])
def test_relative_manifest_paths_are_scoped_and_credential_names_rejected(name):
    with pytest.raises(ValueError):
        module._relative(name)


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unsupported", "foreign_keys", "too_large", "existing"])
def test_source_and_capacity_failures_never_publish_manifest(project, failure):
    root, path, out = project
    sources = (path,)
    options = {"min_free_bytes": 0}
    if failure == "missing":
        sources = (root / "data/research/missing.sqlite",)
    elif failure == "duplicate":
        sources = (path, path)
    elif failure == "unsupported":
        with closing(sqlite3.connect(path)) as db:
            db.execute("CREATE TABLE arbitrary_credentials(value TEXT)")
    elif failure == "foreign_keys":
        with closing(sqlite3.connect(path)) as db:
            db.execute("INSERT INTO opportunity_sources VALUES ('absent')")
            db.commit()
    elif failure == "too_large":
        options["max_bytes"] = 1
    else:
        out.mkdir(parents=True)
    with pytest.raises(ValueError):
        module.create(root, sources, out, **options)
    assert not (out / "manifest.json").exists()
    assert not (root / "data/research/missing.sqlite").exists()


def test_disk_reserve_is_checked_before_mutation(project, monkeypatch):
    root, path, out = project
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 99, 1))
    with pytest.raises(OSError, match="CAPACITY"):
        module.create(root, (path,), out, min_free_bytes=10)
    assert not out.exists()


def test_backup_busy_deadline_is_finite_and_partial_not_verified(project, monkeypatch):
    root, path, out = project
    ticks = [0]
    original_connect = sqlite3.connect
    class BusySource(sqlite3.Connection):
        def backup(self, target, **kwargs):
            ticks[0] = 10
            kwargs["progress"](sqlite3.SQLITE_BUSY, 1, 1)
    def connect(database, **kwargs):
        if str(database) == path.as_uri()+"?mode=ro":
            kwargs["factory"] = BusySource
        return original_connect(database, **kwargs)
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    monkeypatch.setattr(module, "monotonic", lambda: ticks[0])
    with pytest.raises(TimeoutError):
        module.create(root, (path,), out, min_free_bytes=0, timeout_seconds=1)
    assert not (out / "manifest.json").exists() and (out / "failure.json").is_file()


@pytest.mark.parametrize("change", ["bytes", "missing", "extra", "wal", "manifest", "wrong_hash", "counts", "schema"])
def test_pinned_manifest_and_snapshot_corruption_rejected(project, change):
    _, _, out = project
    result, manifest = create(project)
    sha = result["manifest_sha256"]
    snapshot = out / "database-00.sqlite"
    if change == "bytes":
        with snapshot.open("r+b") as f:
            f.seek(-1, 2)
            f.write(b"x")
    elif change == "missing":
        snapshot.unlink()
    elif change in ("extra", "wal"):
        (out / ("unlisted.txt" if change == "extra" else "database-00.sqlite-wal")).write_text("unknown")
    elif change == "wrong_hash":
        sha = "0"*64
    else:
        if change == "manifest":
            manifest["total_bytes"] += 1
        elif change == "counts":
            manifest["databases"][0]["audit"]["row_counts"]["opportunities"] += 1
        else:
            manifest["databases"][0]["audit"]["schema_sha256"] = "0"*64
        (out / "manifest.json").write_text(exact_json(manifest), encoding="utf-8")
        if change != "manifest":
            sha = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module.verify(out, sha)


@pytest.mark.parametrize("name,value", [("max_bytes", True), ("max_bytes", 0), ("min_free_bytes", -1),
    ("timeout_seconds", 0), ("timeout_seconds", 301)])
def test_finite_typed_limits_required(project, name, value):
    root, path, out = project
    kwargs = {"max_bytes": 1024**2, "min_free_bytes": 0, "timeout_seconds": 10, name: value}
    with pytest.raises(ValueError):
        module.create(root, (path,), out, **kwargs)
    assert not out.exists()


def test_rehearsal_never_overwrites_original_or_existing_archive(project):
    root, path, out = project
    result, _ = create(project)
    original = path.read_bytes()
    for target in (root, path.parent, out, out / "nested", root / "data/research/restore"):
        with pytest.raises(ValueError):
            module.rehearse(root, out, result["manifest_sha256"], target)
    assert path.read_bytes() == original


def test_cli_create_verify_rehearse_and_unavailable_are_local(project, monkeypatch, capsys):
    root, path, out = project
    monkeypatch.chdir(root)
    assert module.main(["create", "--database", str(path), "--output-dir", str(out)]) == 0
    result = json.loads(capsys.readouterr().out)
    options = ["--backup-dir", str(out), "--manifest-sha256", result["manifest_sha256"]]
    assert module.main(["verify", *options]) == 0
    assert module.main(["rehearse", *options, "--output-dir", str(root / "backups/rehearsal")]) == 0
    assert module.main(["verify", "--backup-dir", str(out), "--manifest-sha256", "0"*64]) == 1
    assert "UNAVAILABLE" in capsys.readouterr().err


@pytest.mark.parametrize("change", ["future", "naive", "reversed", "duplicate_json", "false_bool", "over_cap"])
def test_manifest_chronology_and_exact_structure_are_verified(project, change):
    _, _, out = project
    result, manifest = create(project)
    if change == "future":
        manifest["created_at"] = "2100-01-01T00:00:00+00:00"
    elif change == "naive":
        manifest["databases"][0]["snapshot_started_at"] = "2020-01-01T00:00:00"
    elif change == "reversed":
        manifest["databases"][0]["snapshot_finished_at"] = "2020-01-01T00:00:00+00:00"
    elif change == "false_bool":
        manifest["credentials_included"] = 0
    elif change == "over_cap":
        manifest["max_bytes"] = 1
    body = exact_json(manifest)
    if change == "duplicate_json":
        body = body[:-1]+',"schema_version":1}'
    (out / "manifest.json").write_text(body, encoding="utf-8")
    sha = hashlib.sha256((out / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        module.verify(out, sha)


def test_credential_file_not_opened_or_copied(project, monkeypatch):
    root, _, out = project
    path = root / "data/testnet/credentials.dpapi"
    path.parent.mkdir()
    path.write_bytes(b"do not read")
    connect = Mock(side_effect=AssertionError("no credential access"))
    monkeypatch.setattr(module.sqlite3, "connect", connect)
    with pytest.raises(ValueError):
        module.create(root, (path,), out)
    connect.assert_not_called()
    assert not out.exists()


def test_restore_capacity_failure_leaves_original_and_target_untouched(project, monkeypatch):
    root, path, out = project
    result, _ = create(project)
    before = path.read_bytes()
    monkeypatch.setattr(module.shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(100, 99, 1))
    target = root / "backups/rehearsal"
    with pytest.raises(OSError, match="CAPACITY"):
        module.rehearse(root, out, result["manifest_sha256"], target)
    assert not target.exists() and path.read_bytes() == before


def test_multiple_database_snapshots_are_separately_attributed(project):
    root, path, out = project
    other = path.with_name("second.sqlite")
    shutil.copyfile(path, other)
    result = module.create(root, (path, other), out, min_free_bytes=0)
    manifest = module.verify(out, result["manifest_sha256"])
    assert len(manifest["databases"]) == 2 and result["databases"] == 2
    assert manifest["global_atomic_snapshot"] is False
    assert manifest["databases"][0]["source_relative_path"] != manifest["databases"][1]["source_relative_path"]
