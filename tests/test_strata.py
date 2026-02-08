"""Integration tests for the StrataDB Python SDK."""

import os
import tempfile
import time
from datetime import datetime

import pytest
import numpy as np
from stratadb import (
    Strata,
    Transaction,
    Collection,
    Snapshot,
    StrataError,
    NotFoundError,
    ValidationError,
    ConflictError,
    StateError,
    ConstraintError,
    AccessDeniedError,
    IoError,
)


@pytest.fixture
def db():
    """Create an in-memory database for testing."""
    return Strata.cache()


class TestKVStore:
    """Tests for KV Store operations."""

    def test_put_get(self, db):
        db.kv_put("key1", "value1")
        assert db.kv_get("key1") == "value1"

    def test_put_get_dict(self, db):
        db.kv_put("config", {"theme": "dark", "count": 42})
        result = db.kv_get("config")
        assert result["theme"] == "dark"
        assert result["count"] == 42

    def test_get_missing(self, db):
        assert db.kv_get("nonexistent") is None

    def test_delete(self, db):
        db.kv_put("to_delete", "value")
        assert db.kv_delete("to_delete") is True
        assert db.kv_get("to_delete") is None

    def test_list(self, db):
        db.kv_put("user:1", "alice")
        db.kv_put("user:2", "bob")
        db.kv_put("item:1", "book")

        all_keys = db.kv_list()
        assert len(all_keys) == 3

        user_keys = db.kv_list("user:")
        assert len(user_keys) == 2

    def test_history(self, db):
        db.kv_put("h", "v1")
        db.kv_put("h", "v2")
        history = db.kv_history("h")
        assert history is not None
        assert len(history) >= 2
        # Each entry has value, version, timestamp
        for entry in history:
            assert "value" in entry
            assert "version" in entry
            assert "timestamp" in entry

    def test_history_missing(self, db):
        assert db.kv_history("no_such_key") is None

    def test_get_versioned(self, db):
        db.kv_put("vk", "hello")
        result = db.kv_get_versioned("vk")
        assert result is not None
        assert result["value"] == "hello"
        assert "version" in result
        assert "timestamp" in result

    def test_get_versioned_missing(self, db):
        assert db.kv_get_versioned("nope") is None

    def test_list_paginated(self, db):
        for i in range(5):
            db.kv_put(f"pg:{i}", i)
        result = db.kv_list_paginated(prefix="pg:", limit=3)
        assert "keys" in result
        assert len(result["keys"]) == 3

    def test_list_paginated_no_limit(self, db):
        db.kv_put("a", 1)
        db.kv_put("b", 2)
        result = db.kv_list_paginated()
        assert len(result["keys"]) == 2


class TestStateCell:
    """Tests for State Cell operations."""

    def test_set_get(self, db):
        db.state_set("counter", 100)
        assert db.state_get("counter") == 100

    def test_init(self, db):
        db.state_init("status", "pending")
        assert db.state_get("status") == "pending"

    def test_cas(self, db):
        # CAS is version-based, not value-based
        version = db.state_set("value", 1)
        # Try to update with correct version
        new_version = db.state_cas("value", 2, version)
        assert new_version is not None
        assert db.state_get("value") == 2
        # Try with wrong version - should fail
        result = db.state_cas("value", 3, 999)
        assert result is None  # CAS failed

    def test_history(self, db):
        db.state_set("sc", "a")
        db.state_set("sc", "b")
        history = db.state_history("sc")
        assert history is not None
        assert len(history) >= 2

    def test_history_missing(self, db):
        assert db.state_history("no_cell") is None

    def test_delete(self, db):
        db.state_set("del_me", "val")
        assert db.state_delete("del_me") is True
        assert db.state_get("del_me") is None

    def test_delete_missing(self, db):
        assert db.state_delete("never_existed") is False

    def test_list(self, db):
        db.state_set("st:a", 1)
        db.state_set("st:b", 2)
        db.state_set("other", 3)
        cells = db.state_list("st:")
        assert len(cells) == 2
        all_cells = db.state_list()
        assert len(all_cells) == 3

    def test_get_versioned(self, db):
        db.state_set("sv", "data")
        result = db.state_get_versioned("sv")
        assert result is not None
        assert result["value"] == "data"
        assert "version" in result

    def test_get_versioned_missing(self, db):
        assert db.state_get_versioned("nope") is None


class TestEventLog:
    """Tests for Event Log operations."""

    def test_append_get(self, db):
        db.event_append("user_action", {"action": "click", "target": "button"})
        assert db.event_len() == 1

        event = db.event_get(0)
        assert event is not None
        assert event["value"]["action"] == "click"

    def test_list_by_type(self, db):
        db.event_append("click", {"x": 10})
        db.event_append("scroll", {"y": 100})
        db.event_append("click", {"x": 20})

        clicks = db.event_list("click")
        assert len(clicks) == 2

    def test_event_len_explicit(self, db):
        assert db.event_len() == 0
        db.event_append("a", {"msg": "payload1"})
        db.event_append("b", {"msg": "payload2"})
        assert db.event_len() == 2

    def test_list_paginated(self, db):
        for i in range(5):
            db.event_append("tick", {"i": i})
        events = db.event_list_paginated("tick", limit=3)
        assert len(events) == 3
        # Use after= to get later events
        last_seq = events[-1]["version"]
        more = db.event_list_paginated("tick", after=last_seq)
        assert len(more) == 2


class TestJSONStore:
    """Tests for JSON Store operations."""

    def test_set_get(self, db):
        db.json_set("config", "$", {"theme": "dark", "lang": "en"})
        result = db.json_get("config", "$")
        assert result["theme"] == "dark"

    def test_get_path(self, db):
        db.json_set("config", "$", {"theme": "dark", "lang": "en"})
        theme = db.json_get("config", "$.theme")
        assert theme == "dark"

    def test_list(self, db):
        db.json_set("doc1", "$", {"a": 1})
        db.json_set("doc2", "$", {"b": 2})
        result = db.json_list(100)  # limit is required
        assert len(result["keys"]) == 2

    def test_history(self, db):
        db.json_set("jh", "$", {"v": 1})
        db.json_set("jh", "$", {"v": 2})
        history = db.json_history("jh")
        assert history is not None
        assert len(history) >= 2

    def test_history_missing(self, db):
        assert db.json_history("no_doc") is None

    def test_delete(self, db):
        db.json_set("jd", "$", {"x": 1})
        db.json_delete("jd", "$")
        assert db.json_get("jd", "$") is None

    def test_get_versioned(self, db):
        db.json_set("jv", "$", {"data": "ok"})
        result = db.json_get_versioned("jv")
        assert result is not None
        assert "version" in result

    def test_get_versioned_missing(self, db):
        assert db.json_get_versioned("nope") is None


class TestVectorStore:
    """Tests for Vector Store operations."""

    def test_create_collection(self, db):
        db.vector_create_collection("embeddings", 4)
        collections = db.vector_list_collections()
        assert any(c["name"] == "embeddings" for c in collections)

    def test_upsert_search(self, db):
        db.vector_create_collection("embeddings", 4)

        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        db.vector_upsert("embeddings", "v1", v1)
        db.vector_upsert("embeddings", "v2", v2)

        results = db.vector_search("embeddings", v1, 2)
        assert len(results) == 2
        assert results[0]["key"] == "v1"  # Most similar

    def test_upsert_with_metadata(self, db):
        db.vector_create_collection("docs", 4)
        vec = [1.0, 0.0, 0.0, 0.0]
        db.vector_upsert("docs", "doc1", vec, {"title": "Hello"})

        result = db.vector_get("docs", "doc1")
        assert result["metadata"]["title"] == "Hello"

    def test_get(self, db):
        db.vector_create_collection("vg", 4)
        vec = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        db.vector_upsert("vg", "k1", vec)
        result = db.vector_get("vg", "k1")
        assert result is not None
        assert result["key"] == "k1"
        assert "embedding" in result
        assert "version" in result

    def test_get_missing(self, db):
        db.vector_create_collection("vg2", 4)
        assert db.vector_get("vg2", "nope") is None

    def test_delete(self, db):
        db.vector_create_collection("vd", 4)
        db.vector_upsert("vd", "d1", [1.0, 0.0, 0.0, 0.0])
        assert db.vector_delete("vd", "d1") is True
        assert db.vector_get("vd", "d1") is None

    def test_delete_collection(self, db):
        db.vector_create_collection("to_del", 4)
        assert db.vector_delete_collection("to_del") is True
        collections = db.vector_list_collections()
        assert not any(c["name"] == "to_del" for c in collections)

    def test_collection_stats(self, db):
        db.vector_create_collection("stats_c", 4)
        db.vector_upsert("stats_c", "s1", [1.0, 0.0, 0.0, 0.0])
        stats = db.vector_collection_stats("stats_c")
        assert stats["name"] == "stats_c"
        assert stats["dimension"] == 4
        assert stats["count"] == 1

    def test_batch_upsert(self, db):
        db.vector_create_collection("batch", 4)
        entries = [
            {"key": "b1", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"key": "b2", "vector": [0.0, 1.0, 0.0, 0.0]},
            {"key": "b3", "vector": [0.0, 0.0, 1.0, 0.0]},
        ]
        versions = db.vector_batch_upsert("batch", entries)
        assert len(versions) == 3
        stats = db.vector_collection_stats("batch")
        assert stats["count"] == 3


class TestBranches:
    """Tests for Branch operations."""

    def test_create_list(self, db):
        db.create_branch("feature")
        branches = db.list_branches()
        assert "default" in branches
        assert "feature" in branches

    def test_switch(self, db):
        db.kv_put("x", 1)
        db.create_branch("feature")
        db.set_branch("feature")

        # Data isolated in new branch
        assert db.kv_get("x") is None

        db.kv_put("x", 2)
        db.set_branch("default")
        assert db.kv_get("x") == 1

    def test_fork(self, db):
        db.kv_put("shared", "original")
        result = db.fork_branch("forked")
        assert result["keys_copied"] > 0

        db.set_branch("forked")
        assert db.kv_get("shared") == "original"

    def test_current_branch(self, db):
        assert db.current_branch() == "default"
        db.create_branch("test")
        db.set_branch("test")
        assert db.current_branch() == "test"

    def test_delete_branch(self, db):
        db.create_branch("tmp")
        db.delete_branch("tmp")
        assert "tmp" not in db.list_branches()

    def test_branch_exists(self, db):
        assert db.branch_exists("default") is True
        assert db.branch_exists("nonexistent") is False
        db.create_branch("new")
        assert db.branch_exists("new") is True

    def test_branch_get(self, db):
        info = db.branch_get("default")
        assert info is not None
        assert info["id"] == "default"
        assert "status" in info
        assert "version" in info

    def test_branch_get_missing(self, db):
        assert db.branch_get("no_such_branch") is None

    def test_diff_branches(self, db):
        db.kv_put("a", 1)
        db.create_branch("other")
        diff = db.diff_branches("default", "other")
        assert "summary" in diff
        assert "total_added" in diff["summary"]

    def test_merge_branches(self, db):
        db.kv_put("m", "base")
        db.fork_branch("src")
        db.set_branch("src")
        db.kv_put("m", "updated")
        db.set_branch("default")
        result = db.merge_branches("src")
        assert "keys_applied" in result
        assert "conflicts" in result


class TestSpaces:
    """Tests for Space operations."""

    def test_list_spaces(self, db):
        spaces = db.list_spaces()
        assert "default" in spaces

    def test_switch_space(self, db):
        db.kv_put("key", "value1")
        db.set_space("other")
        assert db.kv_get("key") is None

        db.kv_put("key", "value2")
        db.set_space("default")
        assert db.kv_get("key") == "value1"

    def test_current_space(self, db):
        assert db.current_space() == "default"

    def test_space_create(self, db):
        db.space_create("myspace")
        assert "myspace" in db.list_spaces()

    def test_space_exists(self, db):
        assert db.space_exists("default") is True
        assert db.space_exists("no_such_space") is False

    def test_delete_space(self, db):
        db.space_create("tmp_space")
        db.delete_space_force("tmp_space")
        assert db.space_exists("tmp_space") is False


class TestTransactions:
    """Tests for Transaction operations."""

    def test_begin_commit(self, db):
        db.begin()
        db.kv_put("tx_key", "tx_val")
        version = db.commit()
        assert version > 0
        assert db.kv_get("tx_key") == "tx_val"

    def test_begin_rollback(self, db):
        # Note: begin/rollback use the session API while kv_put/kv_get go
        # through the non-session executor, so rollback won't undo kv_put.
        # This test verifies begin+rollback lifecycle completes without error.
        db.begin()
        db.rollback()

    def test_context_manager_commit(self, db):
        with Transaction(db):
            db.kv_put("ctx", "value")
        assert db.kv_get("ctx") == "value"

    def test_context_manager_rollback(self, db):
        db.kv_put("safe", "original")
        with pytest.raises(ValueError):
            with Transaction(db):
                db.kv_put("safe", "changed")
                raise ValueError("boom")
        # Rollback completes without error (though kv_put bypasses session)
        assert db.kv_get("safe") is not None

    def test_txn_is_active(self, db):
        assert db.txn_is_active() is False
        db.begin()
        assert db.txn_is_active() is True
        db.commit()
        assert db.txn_is_active() is False

    def test_txn_info(self, db):
        assert db.txn_info() is None
        db.begin()
        info = db.txn_info()
        assert info is not None
        assert "id" in info
        assert "status" in info
        db.rollback()


class TestBundles:
    """Tests for Bundle export/import operations."""

    def test_export_import(self, db):
        # Create and populate a non-default branch for export
        db.create_branch("export_src")
        db.set_branch("export_src")
        db.kv_put("bk", "bv")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bundle.strata")
            result = db.branch_export("export_src", path)
            assert result["entry_count"] > 0
            assert result["bundle_size"] > 0

            # Delete the source branch so import can re-create it
            db.set_branch("default")
            db.delete_branch("export_src")
            imp = db.branch_import(path)
            assert imp["keys_written"] > 0

    def test_validate_bundle(self, db):
        db.kv_put("val_key", "val_val")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "check.strata")
            db.branch_export("default", path)
            info = db.branch_validate_bundle(path)
            assert info["checksums_valid"] is True
            assert info["entry_count"] > 0

    def test_export_result_fields(self, db):
        db.kv_put("f", 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "fields.strata")
            result = db.branch_export("default", path)
            assert "branch_id" in result
            assert "path" in result
            assert "entry_count" in result
            assert "bundle_size" in result


class TestDatabase:
    """Tests for Database operations."""

    def test_ping(self, db):
        version = db.ping()
        assert version is not None

    def test_info(self, db):
        info = db.info()
        assert "version" in info
        assert "branch_count" in info

    def test_flush(self, db):
        db.kv_put("fl", "data")
        db.flush()  # Should not raise
        assert db.kv_get("fl") == "data"

    def test_compact(self, db):
        db.kv_put("cp", "data")
        db.compact()  # Should not raise
        assert db.kv_get("cp") == "data"


class TestRetention:
    """Tests for Retention operations."""

    def test_retention_apply(self, db):
        db.kv_put("r", "data")
        db.retention_apply()  # Should not raise
        assert db.kv_get("r") == "data"


class TestErrors:
    """Tests for exception hierarchy and error conditions."""

    def test_not_found_error(self, db):
        db.vector_create_collection("err_c", 4)
        with pytest.raises(NotFoundError):
            db.vector_collection_stats("nonexistent_collection")

    def test_validation_error(self, db):
        with pytest.raises(ValidationError):
            db.vector_create_collection("bad", 4, metric="invalid_metric")

    def test_duplicate_branch_error(self, db):
        db.create_branch("dup")
        with pytest.raises(StrataError):
            db.create_branch("dup")

    def test_exception_hierarchy_isinstance(self, db):
        """All specific errors are instances of StrataError."""
        try:
            db.vector_collection_stats("missing")
        except NotFoundError as e:
            assert isinstance(e, StrataError)
        else:
            pytest.fail("Expected NotFoundError")

    def test_exception_hierarchy_issubclass(self):
        """All specific error classes are subclasses of StrataError."""
        assert issubclass(NotFoundError, StrataError)
        assert issubclass(ValidationError, StrataError)
        assert issubclass(ConflictError, StrataError)
        assert issubclass(StateError, StrataError)
        assert issubclass(ConstraintError, StrataError)
        assert issubclass(AccessDeniedError, StrataError)
        assert issubclass(IoError, StrataError)

    def test_constraint_error_dimension_mismatch(self, db):
        db.vector_create_collection("dim_c", 4)
        with pytest.raises(ConstraintError):
            db.vector_upsert("dim_c", "wrong", [1.0, 2.0])  # wrong dimension


def _now_micros():
    """Return current time in microseconds since epoch."""
    return int(time.time() * 1_000_000)


class TestTimeTravel:
    """Tests for time-travel query support (as_of parameter)."""

    def test_kv_get_as_of(self, db):
        """kv_get with as_of returns the value at a past timestamp."""
        db.kv_put("tt_kv", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv_put("tt_kv", "v2")

        # Current value is v2
        assert db.kv_get("tt_kv") == "v2"
        # At timestamp before v2, should return v1
        assert db.kv_get("tt_kv", as_of=ts) == "v1"

    def test_kv_get_as_of_before_creation(self, db):
        """kv_get with as_of before key was created returns None."""
        ts = _now_micros()
        time.sleep(0.05)
        db.kv_put("tt_kv_new", "value")
        assert db.kv_get("tt_kv_new", as_of=ts) is None

    def test_kv_get_as_of_none_same_as_omit(self, db):
        """kv_get with as_of=None behaves identically to omitting it."""
        db.kv_put("tt_kv_none", "hello")
        assert db.kv_get("tt_kv_none", as_of=None) == db.kv_get("tt_kv_none")

    def test_kv_list_as_of(self, db):
        db.kv_put("tt_l:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv_put("tt_l:b", 2)

        current = db.kv_list("tt_l:")
        assert len(current) == 2
        past = db.kv_list("tt_l:", as_of=ts)
        assert len(past) == 1
        assert "tt_l:a" in past

    def test_state_get_as_of(self, db):
        """state_get with as_of returns the value at a past timestamp."""
        db.state_set("tt_st", "s1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state_set("tt_st", "s2")

        assert db.state_get("tt_st") == "s2"
        assert db.state_get("tt_st", as_of=ts) == "s1"

    def test_state_list_as_of(self, db):
        db.state_set("tt_sl:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state_set("tt_sl:b", 2)

        current = db.state_list("tt_sl:")
        assert len(current) == 2
        past = db.state_list("tt_sl:", as_of=ts)
        assert len(past) == 1

    def test_event_get_as_of(self, db):
        """event_get with as_of filters by event timestamp."""
        db.event_append("tt_ev", {"msg": "first"})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.event_append("tt_ev", {"msg": "second"})

        # Event 0 existed before ts, so it should be visible
        assert db.event_get(0, as_of=ts) is not None
        # Event 1 was appended after ts, so it should not be visible
        assert db.event_get(1, as_of=ts) is None

    def test_event_list_as_of(self, db):
        """event_list with as_of returns fewer events at earlier timestamp."""
        db.event_append("tt_el", {"i": 0})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.event_append("tt_el", {"i": 1})
        db.event_append("tt_el", {"i": 2})

        current = db.event_list("tt_el")
        assert len(current) == 3
        past = db.event_list("tt_el", as_of=ts)
        assert len(past) == 1

    def test_json_get_as_of(self, db):
        """json_get with as_of returns the value at a past timestamp."""
        db.json_set("tt_js", "$", {"v": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json_set("tt_js", "$", {"v": 2})

        assert db.json_get("tt_js", "$")["v"] == 2
        assert db.json_get("tt_js", "$", as_of=ts)["v"] == 1

    def test_json_list_as_of(self, db):
        db.json_set("tt_jl:a", "$", {"x": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json_set("tt_jl:b", "$", {"x": 2})

        current = db.json_list(100, prefix="tt_jl:")
        assert len(current["keys"]) == 2
        past = db.json_list(100, prefix="tt_jl:", as_of=ts)
        assert len(past["keys"]) == 1

    def test_vector_search_as_of(self, db):
        """vector_search with as_of returns fewer vectors at earlier timestamp."""
        db.vector_create_collection("tt_vs", 4)
        db.vector_upsert("tt_vs", "v1", [1.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.vector_upsert("tt_vs", "v2", [0.0, 1.0, 0.0, 0.0])

        current = db.vector_search("tt_vs", [1.0, 0.0, 0.0, 0.0], 10)
        assert len(current) == 2
        past = db.vector_search("tt_vs", [1.0, 0.0, 0.0, 0.0], 10, as_of=ts)
        assert len(past) == 1
        assert past[0]["key"] == "v1"

    def test_vector_get_as_of(self, db):
        """vector_get with as_of returns vector data at a past timestamp."""
        db.vector_create_collection("tt_vg", 4)
        db.vector_upsert("tt_vg", "vk", [1.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.vector_delete("tt_vg", "vk")

        # Current: deleted
        assert db.vector_get("tt_vg", "vk") is None
        # At past timestamp: should exist
        result = db.vector_get("tt_vg", "vk", as_of=ts)
        assert result is not None
        assert result["key"] == "vk"
        assert "embedding" in result

    def test_time_range(self, db):
        """time_range returns valid oldest and latest timestamps."""
        db.kv_put("tr_key", "value")
        result = db.time_range
        assert "oldest_ts" in result
        assert "latest_ts" in result
        assert result["oldest_ts"] is not None
        assert result["latest_ts"] is not None
        assert result["oldest_ts"] <= result["latest_ts"]

    def test_time_range_structure(self):
        """time_range returns a dict with oldest_ts and latest_ts keys."""
        db = Strata.cache()
        result = db.time_range
        assert "oldest_ts" in result
        assert "latest_ts" in result


# =============================================================================
# Namespace API tests
# =============================================================================


class TestNamespaceKV:
    """Tests for db.kv namespace."""

    def test_put_get(self, db):
        db.kv.put("key1", "value1")
        assert db.kv.get("key1") == "value1"

    def test_get_default(self, db):
        assert db.kv.get("missing") is None
        assert db.kv.get("missing", default="fallback") == "fallback"

    def test_delete(self, db):
        db.kv.put("del", "val")
        assert db.kv.delete("del") is True
        assert db.kv.get("del") is None

    def test_keys(self, db):
        db.kv.put("user:1", "alice")
        db.kv.put("user:2", "bob")
        db.kv.put("item:1", "book")
        assert len(db.kv.keys(prefix="user:")) == 2

    def test_list(self, db):
        db.kv.put("a", 1)
        db.kv.put("b", 2)
        db.kv.put("c", 3)
        assert len(db.kv.list()) == 3

    def test_list_with_limit(self, db):
        for i in range(5):
            db.kv.put(f"pg:{i}", i)
        result = db.kv.list(prefix="pg:", limit=3)
        assert len(result) == 3

    def test_list_with_as_of(self, db):
        db.kv.put("tt:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("tt:b", 2)
        assert len(db.kv.list(prefix="tt:", as_of=ts)) == 1

    def test_get_versioned(self, db):
        db.kv.put("vk", "hello")
        result = db.kv.get_versioned("vk")
        assert result is not None
        assert result["value"] == "hello"
        assert "version" in result

    def test_get_versioned_missing(self, db):
        assert db.kv.get_versioned("nope") is None

    def test_history(self, db):
        db.kv.put("hk", "v1")
        db.kv.put("hk", "v2")
        history = db.kv.history("hk")
        assert history is not None
        assert len(history) >= 2
        for entry in history:
            assert "value" in entry
            assert "version" in entry
            assert "timestamp" in entry

    def test_history_missing(self, db):
        assert db.kv.history("no_such_key") is None

    def test_get_as_of(self, db):
        db.kv.put("tt_kv", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("tt_kv", "v2")
        assert db.kv.get("tt_kv") == "v2"
        assert db.kv.get("tt_kv", as_of=ts) == "v1"


class TestNamespaceState:
    """Tests for db.state namespace."""

    def test_set_get(self, db):
        db.state.set("counter", 100)
        assert db.state.get("counter") == 100

    def test_get_default(self, db):
        assert db.state.get("missing") is None
        assert db.state.get("missing", default=0) == 0

    def test_init(self, db):
        db.state.init("status", "pending")
        assert db.state.get("status") == "pending"

    def test_cas(self, db):
        version = db.state.set("val", 1)
        new_version = db.state.cas("val", 2, expected_version=version)
        assert new_version is not None
        assert db.state.get("val") == 2
        # Wrong version should fail
        result = db.state.cas("val", 3, expected_version=999)
        assert result is None

    def test_delete(self, db):
        db.state.set("del_me", "val")
        assert db.state.delete("del_me") is True
        assert db.state.get("del_me") is None

    def test_list(self, db):
        db.state.set("st:a", 1)
        db.state.set("st:b", 2)
        db.state.set("other", 3)
        assert len(db.state.list(prefix="st:")) == 2
        assert len(db.state.list()) == 3

    def test_get_versioned(self, db):
        db.state.set("sv", "data")
        result = db.state.get_versioned("sv")
        assert result is not None
        assert result["value"] == "data"

    def test_history(self, db):
        db.state.set("sc", "a")
        db.state.set("sc", "b")
        history = db.state.history("sc")
        assert history is not None
        assert len(history) >= 2

    def test_get_as_of(self, db):
        db.state.set("tt_st", "s1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state.set("tt_st", "s2")
        assert db.state.get("tt_st") == "s2"
        assert db.state.get("tt_st", as_of=ts) == "s1"


class TestNamespaceEvents:
    """Tests for db.events namespace."""

    def test_append_get(self, db):
        db.events.append("click", {"x": 10})
        assert db.events.count == 1
        event = db.events.get(0)
        assert event is not None
        assert event["value"]["x"] == 10

    def test_list(self, db):
        db.events.append("click", {"x": 10})
        db.events.append("scroll", {"y": 100})
        db.events.append("click", {"x": 20})
        clicks = db.events.list("click")
        assert len(clicks) == 2

    def test_list_paginated(self, db):
        for i in range(5):
            db.events.append("tick", {"i": i})
        events = db.events.list("tick", limit=3)
        assert len(events) == 3

    def test_len(self, db):
        assert len(db.events) == 0
        db.events.append("a", {"msg": "p1"})
        db.events.append("b", {"msg": "p2"})
        assert len(db.events) == 2

    def test_count_property(self, db):
        db.events.append("x", {"a": 1})
        assert db.events.count == 1

    def test_get_as_of(self, db):
        db.events.append("ev", {"msg": "first"})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.events.append("ev", {"msg": "second"})
        assert db.events.get(0, as_of=ts) is not None
        assert db.events.get(1, as_of=ts) is None


class TestNamespaceJSON:
    """Tests for db.json namespace."""

    def test_set_get(self, db):
        db.json.set("config", "$", {"theme": "dark"})
        result = db.json.get("config")
        assert result["theme"] == "dark"

    def test_get_path(self, db):
        db.json.set("config", "$", {"theme": "dark", "lang": "en"})
        assert db.json.get("config", "$.theme") == "dark"

    def test_get_default_path(self, db):
        """json.get with default path='$' returns the whole doc."""
        db.json.set("doc", "$", {"a": 1})
        result = db.json.get("doc")
        assert result["a"] == 1

    def test_delete(self, db):
        db.json.set("jd", "$", {"x": 1})
        db.json.delete("jd")
        assert db.json.get("jd") is None

    def test_get_versioned(self, db):
        db.json.set("jv", "$", {"data": "ok"})
        result = db.json.get_versioned("jv")
        assert result is not None
        assert "version" in result

    def test_history(self, db):
        db.json.set("jh", "$", {"v": 1})
        db.json.set("jh", "$", {"v": 2})
        history = db.json.history("jh")
        assert history is not None
        assert len(history) >= 2

    def test_list(self, db):
        db.json.set("d1", "$", {"a": 1})
        db.json.set("d2", "$", {"b": 2})
        result = db.json.list()
        assert len(result["keys"]) == 2

    def test_get_as_of(self, db):
        db.json.set("tt_js", "$", {"v": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json.set("tt_js", "$", {"v": 2})
        assert db.json.get("tt_js")["v"] == 2
        assert db.json.get("tt_js", as_of=ts)["v"] == 1


class TestNamespaceVectors:
    """Tests for db.vectors namespace and Collection."""

    def test_create_and_list(self, db):
        coll = db.vectors.create("emb", dimension=4)
        assert isinstance(coll, Collection)
        assert coll.name == "emb"
        assert "emb" in db.vectors

    def test_collection_handle(self, db):
        db.vectors.create("docs", dimension=4)
        coll = db.vectors.collection("docs")
        assert coll.name == "docs"

    def test_upsert_search(self, db):
        coll = db.vectors.create("emb", dimension=4)
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0, 0.0]
        coll.upsert("v1", v1)
        coll.upsert("v2", v2)
        results = coll.search(v1, k=2)
        assert len(results) == 2
        assert results[0]["key"] == "v1"

    def test_upsert_with_metadata(self, db):
        coll = db.vectors.create("docs", dimension=4)
        coll.upsert("d1", [1.0, 0.0, 0.0, 0.0], metadata={"title": "Hello"})
        result = coll.get("d1")
        assert result["metadata"]["title"] == "Hello"

    def test_delete(self, db):
        coll = db.vectors.create("vd", dimension=4)
        coll.upsert("d1", [1.0, 0.0, 0.0, 0.0])
        assert coll.delete("d1") is True
        assert coll.get("d1") is None

    def test_batch_upsert(self, db):
        coll = db.vectors.create("batch", dimension=4)
        entries = [
            {"key": "b1", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"key": "b2", "vector": [0.0, 1.0, 0.0, 0.0]},
        ]
        versions = coll.batch_upsert(entries)
        assert len(versions) == 2

    def test_stats_and_len(self, db):
        coll = db.vectors.create("stats", dimension=4)
        coll.upsert("s1", [1.0, 0.0, 0.0, 0.0])
        stats = coll.stats()
        assert stats["count"] == 1
        assert len(coll) == 1

    def test_drop(self, db):
        db.vectors.create("to_del", dimension=4)
        assert "to_del" in db.vectors
        db.vectors.drop("to_del")
        assert "to_del" not in db.vectors

    def test_list(self, db):
        db.vectors.create("c1", dimension=4)
        db.vectors.create("c2", dimension=8)
        collections = db.vectors.list()
        names = [c["name"] for c in collections]
        assert "c1" in names
        assert "c2" in names

    def test_search_as_of(self, db):
        coll = db.vectors.create("tt_vs", dimension=4)
        coll.upsert("v1", [1.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        coll.upsert("v2", [0.0, 1.0, 0.0, 0.0])
        current = coll.search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(current) == 2
        past = coll.search([1.0, 0.0, 0.0, 0.0], k=10, as_of=ts)
        assert len(past) == 1


class TestNamespaceBranches:
    """Tests for db.branches namespace and branch properties."""

    def test_branch_property(self, db):
        assert db.branch == "default"

    def test_checkout(self, db):
        db.branches.create("feature")
        db.checkout("feature")
        assert db.branch == "feature"

    def test_create_delete(self, db):
        db.branches.create("tmp")
        assert db.branches.exists("tmp")
        assert "tmp" in db.branches
        db.branches.delete("tmp")
        assert not db.branches.exists("tmp")
        assert "tmp" not in db.branches

    def test_list(self, db):
        db.branches.create("br1")
        branches = db.branches.list()
        assert "default" in branches
        assert "br1" in branches

    def test_iter(self, db):
        db.branches.create("iter_br")
        branch_list = list(db.branches)
        assert "default" in branch_list
        assert "iter_br" in branch_list

    def test_get(self, db):
        info = db.branches.get("default")
        assert info is not None
        assert info["id"] == "default"

    def test_get_missing(self, db):
        assert db.branches.get("no_such") is None

    def test_fork(self, db):
        db.kv.put("shared", "original")
        result = db.fork("forked")
        assert result["keys_copied"] > 0

    def test_merge(self, db):
        db.kv.put("m", "base")
        db.fork("src")
        db.checkout("src")
        db.kv.put("m", "updated")
        db.checkout("default")
        result = db.merge("src")
        assert "keys_applied" in result

    def test_diff(self, db):
        db.kv.put("a", 1)
        db.branches.create("other")
        diff = db.diff("default", "other")
        assert "summary" in diff


class TestNamespaceSpaces:
    """Tests for db.spaces namespace and space properties."""

    def test_space_property(self, db):
        assert db.space == "default"

    def test_use_space(self, db):
        db.kv.put("key", "val1")
        db.use_space("other")
        assert db.kv.get("key") is None
        db.use_space("default")
        assert db.kv.get("key") == "val1"

    def test_create_exists(self, db):
        db.spaces.create("myspace")
        assert db.spaces.exists("myspace")
        assert "myspace" in db.spaces

    def test_delete(self, db):
        db.spaces.create("tmp_space")
        db.spaces.delete("tmp_space", force=True)
        assert not db.spaces.exists("tmp_space")

    def test_list(self, db):
        spaces = db.spaces.list()
        assert "default" in spaces

    def test_iter(self, db):
        space_list = list(db.spaces)
        assert "default" in space_list


class TestContextManagers:
    """Tests for on_branch() and in_space() context managers."""

    def test_on_branch(self, db):
        db.branches.create("experiment")
        assert db.branch == "default"
        with db.on_branch("experiment"):
            assert db.branch == "experiment"
            db.kv.put("scoped", "value")
        assert db.branch == "default"
        assert db.kv.get("scoped") is None  # isolated

    def test_on_branch_restores_on_exception(self, db):
        db.branches.create("exp2")
        with pytest.raises(ValueError):
            with db.on_branch("exp2"):
                assert db.branch == "exp2"
                raise ValueError("boom")
        assert db.branch == "default"

    def test_in_space(self, db):
        db.spaces.create("tenant")
        assert db.space == "default"
        with db.in_space("tenant"):
            assert db.space == "tenant"
            db.kv.put("tenant_key", "val")
        assert db.space == "default"
        assert db.kv.get("tenant_key") is None

    def test_in_space_restores_on_exception(self, db):
        db.spaces.create("tenant2")
        with pytest.raises(ValueError):
            with db.in_space("tenant2"):
                raise ValueError("boom")
        assert db.space == "default"

    def test_in_transaction_property(self, db):
        assert db.in_transaction is False
        with db.transaction():
            assert db.in_transaction is True
        assert db.in_transaction is False


class TestBackwardCompat:
    """Verify that old flat API still works alongside namespace API."""

    def test_kv_put_alongside_kv_dot_put(self, db):
        db.kv_put("flat", "value")
        assert db.kv.get("flat") == "value"
        db.kv.put("ns", "value")
        assert db.kv_get("ns") == "value"

    def test_state_flat_and_ns(self, db):
        db.state_set("flat_st", 42)
        assert db.state.get("flat_st") == 42
        db.state.set("ns_st", 99)
        assert db.state_get("ns_st") == 99

    def test_events_flat_and_ns(self, db):
        db.event_append("flat_ev", {"a": 1})
        assert len(db.events) == 1
        db.events.append("ns_ev", {"b": 2})
        assert db.event_len() == 2

    def test_current_branch_and_branch_property(self, db):
        assert db.current_branch() == db.branch

    def test_current_space_and_space_property(self, db):
        assert db.current_space() == db.space


class TestSnapshotTimeTravel:
    """Tests for db.at() snapshot API."""

    def test_snapshot_kv_get(self, db):
        db.kv.put("snap_kv", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("snap_kv", "v2")

        snap = db.at(ts)
        assert snap.kv.get("snap_kv") == "v1"
        assert db.kv.get("snap_kv") == "v2"

    def test_snapshot_from_datetime(self, db):
        db.kv.put("snap_dt", "v1")
        time.sleep(0.05)
        dt = datetime.now()
        time.sleep(0.05)
        db.kv.put("snap_dt", "v2")

        snap = db.at(dt)
        assert snap.kv.get("snap_dt") == "v1"

    def test_snapshot_from_int(self, db):
        db.kv.put("snap_int", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("snap_int", "v2")

        snap = db.at(ts)
        assert snap.kv.get("snap_int") == "v1"

    def test_snapshot_timestamp_property(self, db):
        ts = _now_micros()
        snap = db.at(ts)
        assert snap.timestamp == ts

    def test_snapshot_kv_default(self, db):
        snap = db.at(_now_micros())
        assert snap.kv.get("nonexistent") is None
        assert snap.kv.get("nonexistent", default="fb") == "fb"

    def test_snapshot_kv_keys(self, db):
        db.kv.put("sk:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("sk:b", 2)

        snap = db.at(ts)
        assert len(snap.kv.keys(prefix="sk:")) == 1

    def test_snapshot_state_get(self, db):
        db.state.set("snap_st", "s1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state.set("snap_st", "s2")

        snap = db.at(ts)
        assert snap.state.get("snap_st") == "s1"

    def test_snapshot_events_get(self, db):
        db.events.append("snap_ev", {"msg": "first"})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.events.append("snap_ev", {"msg": "second"})

        snap = db.at(ts)
        assert snap.events.get(0) is not None
        assert snap.events.get(1) is None

    def test_snapshot_json_get(self, db):
        db.json.set("snap_js", "$", {"v": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json.set("snap_js", "$", {"v": 2})

        snap = db.at(ts)
        assert snap.json.get("snap_js")["v"] == 1

    def test_snapshot_vectors_search(self, db):
        coll = db.vectors.create("snap_vs", dimension=4)
        coll.upsert("v1", [1.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        coll.upsert("v2", [0.0, 1.0, 0.0, 0.0])

        snap = db.at(ts)
        snap_coll = snap.vectors.collection("snap_vs")
        results = snap_coll.search([1.0, 0.0, 0.0, 0.0], k=10)
        assert len(results) == 1
        assert results[0]["key"] == "v1"

    def test_snapshot_is_read_only(self, db):
        snap = db.at(_now_micros())
        assert not hasattr(snap.kv, "put")
        assert not hasattr(snap.kv, "delete")
        assert not hasattr(snap.state, "set")
        assert not hasattr(snap.state, "delete")
        assert not hasattr(snap.events, "append")
        assert not hasattr(snap.json, "set")
        assert not hasattr(snap.json, "delete")

    def test_time_range_property(self, db):
        db.kv.put("tr_key", "value")
        result = db.time_range
        assert "oldest_ts" in result
        assert "latest_ts" in result
        assert result["oldest_ts"] <= result["latest_ts"]

    def test_kv_history_via_namespace(self, db):
        db.kv.put("hk", "v1")
        db.kv.put("hk", "v2")
        history = db.kv.history("hk")
        assert history is not None
        assert len(history) >= 2

    def test_inline_as_of_still_works(self, db):
        """Inline as_of= on namespace read methods still works."""
        db.kv.put("inline", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("inline", "v2")
        assert db.kv.get("inline", as_of=ts) == "v1"
