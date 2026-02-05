"""Integration tests for the StrataDB Python SDK."""

import pytest
import numpy as np
from stratadb import Strata


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


class TestDatabase:
    """Tests for Database operations."""

    def test_ping(self, db):
        version = db.ping()
        assert version is not None

    def test_info(self, db):
        info = db.info()
        assert "version" in info
        assert "branch_count" in info
