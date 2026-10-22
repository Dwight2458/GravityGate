import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from gravitygate_py.cli import import_rust
from gravitygate_py.config import Accounts, Config, data_dir
from gravitygate_py.registry import resolve
from gravitygate_py.storage import Database, NotFound, StoreError, now_ms


def test_defaults_and_independent_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GRAVITYGATE_PY_DATA_DIR", str(tmp_path / "python"))
    monkeypatch.setenv("GRAVITYGATE_CONFIG_DIR", str(tmp_path / "rust"))
    assert data_dir() == tmp_path / "python"
    assert Config().server.port == 18081
    assert Config().server.max_request_body_bytes == 16777216
    assert not (tmp_path / "rust").exists()


@pytest.mark.parametrize(
    "raw",
    [
        {"server": {"port": 0}},
        {"server": {"max_request_body_bytes": 0}},
        {"verification": {"timeout_secs": 301}},
        {"verification": {"scan_interval_secs": 0}},
        {"upstream": {"endpoints": []}},
        {"upstream": {"endpoints": ["file:///tmp"]}},
        {"accounts": {"strategy": "typo"}},
        {"unknown": True},
    ],
)
def test_config_rejects_invalid(raw):
    with pytest.raises(ValidationError):
        Config.model_validate(raw)


def test_config_env_and_secret_redaction(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[server]\nport=18082\napi_keys=["file-key"]\n', encoding="utf-8")
    monkeypatch.setenv("GG_PORT", "18083")
    monkeypatch.setenv("GG_API_KEYS", "one,two")
    config = Config.load(path)
    assert config.server.port == 18083
    assert config.server.api_keys == ["one", "two"]
    assert "one" not in config.toml(redact=True)
    assert "<redacted>" in config.toml(redact=True)


def test_add_selector_and_redacted_repr(tmp_path):
    database = Database(tmp_path / "db")
    a = database.add("synthetic-secret|project|managed", "a@test")
    assert a.project_id == "project"
    assert a.managed_project_id == "managed"
    assert "synthetic-secret" not in repr(a)
    assert "synthetic-secret" not in json.dumps(a.public())
    assert database.get("0").id == a.id
    assert database.get(a.id[:8]).email == "a@test"
    assert database.get("a@test").id == a.id
    assert database.add("synthetic-secret", "renamed@test").id == a.id
    with pytest.raises(NotFound):
        database.get("missing")
    with pytest.raises(StoreError):
        database.add("")


def test_concurrent_updates_merge_across_connections(tmp_path):
    one = Database(tmp_path / "db")
    two = Database(tmp_path / "db")
    a = one.add("token")

    def increment(i):
        (one if i % 2 else two).update(
            a.id, lambda account: account.state.update(counter=account.state.get("counter", 0) + 1)
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(increment, range(40)))
    assert one.get(a.id).state["counter"] == 40


def test_token_budget_is_atomic_across_process_connections(tmp_path):
    database = Database(tmp_path / "db")
    database.add("token")
    config = Accounts(token_bucket_max=3, token_bucket_refill_per_min=0.0001)
    quota = resolve({"model": "gemini-3.8-flash"}).quota_key

    def select(_):
        return Database(database.path).select(quota, config, set())[0]

    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = list(pool.map(select, range(16)))
    assert sum(a is not None for a in selected) == 3


def test_model_scoped_holds_do_not_block_other_models(tmp_path):
    database = Database(tmp_path / "db")
    a = database.add("token")
    flash = resolve({"model": "gemini-3.8-flash-high"}).quota_key
    pro = resolve({"model": "gemini-3.1-pro"}).quota_key
    database.update(a.id, lambda a: a.state.update(holds={flash: now_ms() + 30000}))
    assert database.select(flash, Accounts(), set())[0] is None
    assert database.select(pro, Accounts(), set())[0] is not None


def test_reset_preserves_inflight_lease(tmp_path):
    database = Database(tmp_path / "db")
    a = database.add("token")
    database.update(
        a.id,
        lambda a: a.state.update(
            verification_required=True, revision=3, lease_id="lease", lease_until=now_ms() + 60000
        ),
    )
    database.update(a.id, lambda a: a.reset_holds())
    a = database.get(a.id)
    assert a.running(now_ms())
    assert a.state["revision"] == 4
    assert a.verification()["state"] == "checking"
    assert not a.state.get("recovered_at")


def test_explicit_import_preserves_holds_and_existing_accounts(tmp_path):
    database = Database(tmp_path / "db")
    a = database.add("one", "keep@test")
    source = tmp_path / "accounts.json"
    source.write_text(
        json.dumps(
            {
                "accounts": [
                    {"refresh_token": "one", "email": "overwrite@test"},
                    {
                        "refresh_token": "two|proj",
                        "email": "new@test",
                        "verification_required": True,
                        "verification_url": "https://accounts.google.com/verify",
                        "verification_check": {"id": "old-lease"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    import_rust(database, source)
    assert database.get(a.id).email == "keep@test"
    new = database.get("new@test")
    assert new.project_id == "proj"
    assert new.state["verification_required"]
    assert not new.running(now_ms())
    source.write_text('{"accounts":[{"refresh_token":"three"},{}]}', encoding="utf-8")
    with pytest.raises(ValueError):
        import_rust(database, source)
    assert len(database.list()) == 2
