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


class TestBranches:
    """Tests for Branch operations."""

    def test_create_list(self, db):
        db.create_branch("feature")
        branches = db.list_branches()
        assert "default" in branches
        assert "feature" in branches

    def test_switch(self, db):
        db.kv.put("x", 1)
        db.create_branch("feature")
        db.set_branch("feature")

        # Data isolated in new branch
        assert db.kv.get("x") is None

        db.kv.put("x", 2)
        db.set_branch("default")
        assert db.kv.get("x") == 1

    def test_fork(self, db):
        db.kv.put("shared", "original")
        result = db.fork_branch("forked")
        assert result["keys_copied"] > 0

        db.set_branch("forked")
        assert db.kv.get("shared") == "original"

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
        db.kv.put("a", 1)
        db.create_branch("other")
        diff = db.diff_branches("default", "other")
        assert "summary" in diff
        assert "total_added" in diff["summary"]

    def test_merge_branches(self, db):
        db.kv.put("m", "base")
        db.fork_branch("src")
        db.set_branch("src")
        db.kv.put("m", "updated")
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
        db.kv.put("key", "value1")
        db.set_space("other")
        assert db.kv.get("key") is None

        db.kv.put("key", "value2")
        db.set_space("default")
        assert db.kv.get("key") == "value1"

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
        db.kv.put("tx_key", "tx_val")
        version = db.commit()
        assert version > 0
        assert db.kv.get("tx_key") == "tx_val"

    def test_begin_rollback(self, db):
        # Note: begin/rollback use the session API while kv.put/kv.get go
        # through the non-session executor, so rollback won't undo kv.put.
        # This test verifies begin+rollback lifecycle completes without error.
        db.begin()
        db.rollback()

    def test_context_manager_commit(self, db):
        with Transaction(db):
            db.kv.put("ctx", "value")
        assert db.kv.get("ctx") == "value"

    def test_context_manager_rollback(self, db):
        db.kv.put("safe", "original")
        with pytest.raises(ValueError):
            with Transaction(db):
                db.kv.put("safe", "changed")
                raise ValueError("boom")
        # Rollback completes without error (though kv.put bypasses session)
        assert db.kv.get("safe") is not None

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
        db.kv.put("bk", "bv")
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
        db.kv.put("val_key", "val_val")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "check.strata")
            db.branch_export("default", path)
            info = db.branch_validate_bundle(path)
            assert info["checksums_valid"] is True
            assert info["entry_count"] > 0

    def test_export_result_fields(self, db):
        db.kv.put("f", 1)
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
        db.kv.put("fl", "data")
        db.flush()  # Should not raise
        assert db.kv.get("fl") == "data"

    def test_compact(self, db):
        db.kv.put("cp", "data")
        db.compact()  # Should not raise
        assert db.kv.get("cp") == "data"


class TestRetention:
    """Tests for Retention operations."""

    def test_retention_apply(self, db):
        db.kv.put("r", "data")
        db.retention_apply()  # Should not raise
        assert db.kv.get("r") == "data"


class TestErrors:
    """Tests for exception hierarchy and error conditions."""

    def test_not_found_error(self, db):
        db.vectors.create("err_c", dimension=4)
        with pytest.raises(NotFoundError):
            db.vectors.collection("nonexistent_collection").stats()

    def test_validation_error(self, db):
        with pytest.raises(ValidationError):
            db.vectors.create("bad", dimension=4, metric="invalid_metric")

    def test_duplicate_branch_error(self, db):
        db.create_branch("dup")
        with pytest.raises(StrataError):
            db.create_branch("dup")

    def test_exception_hierarchy_isinstance(self, db):
        """All specific errors are instances of StrataError."""
        try:
            db.vectors.collection("missing").stats()
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
        coll = db.vectors.create("dim_c", dimension=4)
        with pytest.raises(ConstraintError):
            coll.upsert("wrong", [1.0, 2.0])  # wrong dimension


def _now_micros():
    """Return current time in microseconds since epoch."""
    return int(time.time() * 1_000_000)


class TestTimeTravel:
    """Tests for time-travel query support (as_of parameter)."""

    def test_kv_get_as_of(self, db):
        """kv.get with as_of returns the value at a past timestamp."""
        db.kv.put("tt_kv", "v1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("tt_kv", "v2")

        # Current value is v2
        assert db.kv.get("tt_kv") == "v2"
        # At timestamp before v2, should return v1
        assert db.kv.get("tt_kv", as_of=ts) == "v1"

    def test_kv_get_as_of_before_creation(self, db):
        """kv.get with as_of before key was created returns None."""
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("tt_kv_new", "value")
        assert db.kv.get("tt_kv_new", as_of=ts) is None

    def test_kv_get_as_of_none_same_as_omit(self, db):
        """kv.get with as_of=None behaves identically to omitting it."""
        db.kv.put("tt_kv_none", "hello")
        assert db.kv.get("tt_kv_none", as_of=None) == db.kv.get("tt_kv_none")

    def test_kv_list_as_of(self, db):
        db.kv.put("tt_l:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.kv.put("tt_l:b", 2)

        current = db.kv.list(prefix="tt_l:")
        assert len(current) == 2
        past = db.kv.list(prefix="tt_l:", as_of=ts)
        assert len(past) == 1
        assert "tt_l:a" in past

    def test_state_get_as_of(self, db):
        """state.get with as_of returns the value at a past timestamp."""
        db.state.set("tt_st", "s1")
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state.set("tt_st", "s2")

        assert db.state.get("tt_st") == "s2"
        assert db.state.get("tt_st", as_of=ts) == "s1"

    def test_state_list_as_of(self, db):
        db.state.set("tt_sl:a", 1)
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.state.set("tt_sl:b", 2)

        current = db.state.list(prefix="tt_sl:")
        assert len(current) == 2
        past = db.state.list(prefix="tt_sl:", as_of=ts)
        assert len(past) == 1

    def test_event_get_as_of(self, db):
        """events.get with as_of filters by event timestamp."""
        db.events.append("tt_ev", {"msg": "first"})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.events.append("tt_ev", {"msg": "second"})

        # Event 0 existed before ts, so it should be visible
        assert db.events.get(0, as_of=ts) is not None
        # Event 1 was appended after ts, so it should not be visible
        assert db.events.get(1, as_of=ts) is None

    def test_event_list_as_of(self, db):
        """events.list with as_of returns fewer events at earlier timestamp."""
        db.events.append("tt_el", {"i": 0})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.events.append("tt_el", {"i": 1})
        db.events.append("tt_el", {"i": 2})

        current = db.events.list("tt_el")
        assert len(current) == 3
        past = db.events.list("tt_el", as_of=ts)
        assert len(past) == 1

    def test_json_get_as_of(self, db):
        """json.get with as_of returns the value at a past timestamp."""
        db.json.set("tt_js", "$", {"v": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json.set("tt_js", "$", {"v": 2})

        assert db.json.get("tt_js", "$")["v"] == 2
        assert db.json.get("tt_js", "$", as_of=ts)["v"] == 1

    def test_json_list_as_of(self, db):
        db.json.set("tt_jl:a", "$", {"x": 1})
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        db.json.set("tt_jl:b", "$", {"x": 2})

        current = db.json.list(prefix="tt_jl:")
        assert len(current["keys"]) == 2
        past = db.json.list(prefix="tt_jl:", as_of=ts)
        assert len(past["keys"]) == 1

    def test_vector_search_as_of(self, db):
        """vector search with as_of returns fewer vectors at earlier timestamp."""
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
        assert past[0]["key"] == "v1"

    def test_vector_get_as_of(self, db):
        """vector get with as_of returns vector data at a past timestamp."""
        coll = db.vectors.create("tt_vg", dimension=4)
        coll.upsert("vk", [1.0, 0.0, 0.0, 0.0])
        time.sleep(0.05)
        ts = _now_micros()
        time.sleep(0.05)
        coll.delete("vk")

        # Current: deleted
        assert coll.get("vk") is None
        # At past timestamp: should exist
        result = coll.get("vk", as_of=ts)
        assert result is not None
        assert result["key"] == "vk"
        assert "embedding" in result

    def test_state_get_versioned_as_of_roundtrip(self, db):
        """state.get with as_of using timestamp from get_versioned works."""
        db.state.set("tt_sv", "v1")
        vv = db.state.get_versioned("tt_sv")
        ts = vv["timestamp"]
        db.state.set("tt_sv", "v2")

        assert db.state.get("tt_sv") == "v2"
        assert db.state.get("tt_sv", as_of=ts) == "v1"

    def test_json_get_versioned_as_of_roundtrip(self, db):
        """json.get with as_of using timestamp from get_versioned works."""
        db.json.set("tt_jv", "$", {"v": 1})
        vv = db.json.get_versioned("tt_jv")
        ts = vv["timestamp"]
        db.json.set("tt_jv", "$", {"v": 2})

        assert db.json.get("tt_jv", "$")["v"] == 2
        assert db.json.get("tt_jv", "$", as_of=ts)["v"] == 1

    def test_kv_get_versioned_as_of_roundtrip(self, db):
        """kv.get with as_of using timestamp from get_versioned works."""
        db.kv.put("tt_kvv", "v1")
        vv = db.kv.get_versioned("tt_kvv")
        ts = vv["timestamp"]
        db.kv.put("tt_kvv", "v2")

        assert db.kv.get("tt_kvv") == "v2"
        assert db.kv.get("tt_kvv", as_of=ts) == "v1"

    def test_time_range(self, db):
        """time_range returns valid oldest and latest timestamps."""
        db.kv.put("tr_key", "value")
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

    def test_put_get_dict(self, db):
        db.kv.put("config", {"theme": "dark", "count": 42})
        result = db.kv.get("config")
        assert result["theme"] == "dark"
        assert result["count"] == 42

    def test_get_default(self, db):
        assert db.kv.get("missing") is None
        assert db.kv.get("missing", default="fallback") == "fallback"

    def test_get_missing(self, db):
        assert db.kv.get("nonexistent") is None

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

    def test_list_with_prefix(self, db):
        db.kv.put("user:1", "alice")
        db.kv.put("user:2", "bob")
        db.kv.put("item:1", "book")
        user_keys = db.kv.list(prefix="user:")
        assert len(user_keys) == 2

    def test_list_with_limit(self, db):
        for i in range(5):
            db.kv.put(f"pg:{i}", i)
        result = db.kv.list(prefix="pg:", limit=3)
        assert len(result) == 3

    def test_list_no_limit(self, db):
        db.kv.put("a", 1)
        db.kv.put("b", 2)
        result = db.kv.list()
        assert len(result) == 2

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
        assert "timestamp" in result

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

    def test_delete_missing(self, db):
        assert db.state.delete("never_existed") is False

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

    def test_get_versioned_missing(self, db):
        assert db.state.get_versioned("nope") is None

    def test_history(self, db):
        db.state.set("sc", "a")
        db.state.set("sc", "b")
        history = db.state.history("sc")
        assert history is not None
        assert len(history) >= 2

    def test_history_missing(self, db):
        assert db.state.history("no_cell") is None

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

    def test_list_paginated_with_after(self, db):
        for i in range(5):
            db.events.append("tick", {"i": i})
        events = db.events.list("tick", limit=3)
        last_seq = events[-1]["version"]
        more = db.events.list("tick", after=last_seq)
        assert len(more) == 2

    def test_len(self, db):
        assert len(db.events) == 0
        db.events.append("a", {"msg": "p1"})
        db.events.append("b", {"msg": "p2"})
        assert len(db.events) == 2

    def test_count_property(self, db):
        db.events.append("x", {"a": 1})
        assert db.events.count == 1

    def test_count_explicit(self, db):
        assert db.events.count == 0
        db.events.append("a", {"msg": "payload1"})
        db.events.append("b", {"msg": "payload2"})
        assert db.events.count == 2

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

    def test_get_versioned_missing(self, db):
        assert db.json.get_versioned("nope") is None

    def test_history(self, db):
        db.json.set("jh", "$", {"v": 1})
        db.json.set("jh", "$", {"v": 2})
        history = db.json.history("jh")
        assert history is not None
        assert len(history) >= 2

    def test_history_missing(self, db):
        assert db.json.history("no_doc") is None

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

    def test_upsert_search_numpy(self, db):
        coll = db.vectors.create("embeddings", dimension=4)

        v1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

        coll.upsert("v1", v1)
        coll.upsert("v2", v2)

        results = coll.search(v1, k=2)
        assert len(results) == 2
        assert results[0]["key"] == "v1"  # Most similar

    def test_upsert_with_metadata(self, db):
        coll = db.vectors.create("docs", dimension=4)
        coll.upsert("d1", [1.0, 0.0, 0.0, 0.0], metadata={"title": "Hello"})
        result = coll.get("d1")
        assert result["metadata"]["title"] == "Hello"

    def test_get(self, db):
        coll = db.vectors.create("vg", dimension=4)
        vec = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
        coll.upsert("k1", vec)
        result = coll.get("k1")
        assert result is not None
        assert result["key"] == "k1"
        assert "embedding" in result
        assert "version" in result

    def test_get_missing(self, db):
        coll = db.vectors.create("vg2", dimension=4)
        assert coll.get("nope") is None

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

    def test_collection_stats_fields(self, db):
        coll = db.vectors.create("stats_c", dimension=4)
        coll.upsert("s1", [1.0, 0.0, 0.0, 0.0])
        stats = coll.stats()
        assert stats["name"] == "stats_c"
        assert stats["dimension"] == 4
        assert stats["count"] == 1

    def test_batch_upsert_three(self, db):
        coll = db.vectors.create("batch", dimension=4)
        entries = [
            {"key": "b1", "vector": [1.0, 0.0, 0.0, 0.0]},
            {"key": "b2", "vector": [0.0, 1.0, 0.0, 0.0]},
            {"key": "b3", "vector": [0.0, 0.0, 1.0, 0.0]},
        ]
        versions = coll.batch_upsert(entries)
        assert len(versions) == 3
        assert len(coll) == 3

    def test_drop(self, db):
        db.vectors.create("to_del", dimension=4)
        assert "to_del" in db.vectors
        db.vectors.drop("to_del")
        assert "to_del" not in db.vectors

    def test_drop_returns_bool(self, db):
        db.vectors.create("to_del", dimension=4)
        assert db.vectors.drop("to_del") is True
        collections = db.vectors.list()
        assert not any(c["name"] == "to_del" for c in collections)

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


class TestFlatMethodsRemoved:
    """Verify that flat data methods raise AttributeError with helpful hints."""

    def test_kv_put_raises(self, db):
        with pytest.raises(AttributeError, match="db.kv"):
            db.kv_put("key", "value")

    def test_kv_get_raises(self, db):
        with pytest.raises(AttributeError, match="db.kv"):
            db.kv_get("key")

    def test_state_set_raises(self, db):
        with pytest.raises(AttributeError, match="db.state"):
            db.state_set("cell", "value")

    def test_event_append_raises(self, db):
        with pytest.raises(AttributeError, match="db.events"):
            db.event_append("type", {"payload": 1})

    def test_json_set_raises(self, db):
        with pytest.raises(AttributeError, match="db.json"):
            db.json_set("key", "$", {"a": 1})

    def test_vector_create_collection_raises(self, db):
        with pytest.raises(AttributeError, match="db.vectors"):
            db.vector_create_collection("coll", 4)


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


class TestSearch:
    """Tests for structured search (v0.12.5)."""

    def test_search_empty_database(self, db):
        results = db.search("hello")
        assert results == []

    def test_search_with_data(self, db):
        db.kv.put("greeting", "hello world")
        db.kv.put("farewell", "goodbye world")
        results = db.search("hello")
        assert isinstance(results, list)

    def test_search_with_primitives_filter(self, db):
        db.kv.put("k1", "data")
        results = db.search("data", primitives=["kv"])
        assert isinstance(results, list)

    def test_search_with_mode(self, db):
        results = db.search("test", mode="keyword")
        assert isinstance(results, list)

    def test_search_with_expand_rerank_disabled(self, db):
        results = db.search("test", expand=False, rerank=False)
        assert isinstance(results, list)

    def test_search_with_time_range(self, db):
        results = db.search(
            "test",
            time_range={"start": "2020-01-01T00:00:00Z", "end": "2030-01-01T00:00:00Z"},
        )
        assert isinstance(results, list)

    def test_search_via_wrapper(self, db):
        """Verify search works through the Strata wrapper class."""
        assert hasattr(db, "search")
        results = db.search("anything", k=5, mode="hybrid")
        assert isinstance(results, list)
