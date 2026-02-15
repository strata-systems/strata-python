"""
StrataDB Python SDK

An embedded database for AI agents with six primitives: KV Store, Event Log,
State Cell, JSON Store, Vector Store, and Branches.

Example usage::

    from stratadb import Strata

    db = Strata.open("/data")

    # Pythonic namespace API
    db.kv.put("user:123", "Alice")
    print(db.kv.get("user:123"))

    # Vector search
    import numpy as np
    coll = db.vectors.create("docs", dimension=384)
    coll.upsert("doc-1", np.random.rand(384).astype(np.float32))
    results = coll.search(np.random.rand(384).astype(np.float32), k=5)

    # Branch operations
    print(db.branch)                    # current branch name
    db.checkout("feature")
    with db.on_branch("experiment"):
        db.kv.put("temp", "value")

    # Transactions
    with db.transaction():
        db.kv.put("a", 1)
        db.kv.put("b", 2)

    # Time-travel snapshots
    from datetime import datetime
    snapshot = db.at(datetime(2024, 6, 15, 12, 0))
    snapshot.kv.get("user:123")         # reads value as of that timestamp

Limitations:
    - **Bytes roundtrip via bundles:** ``Value.Bytes`` (Python ``bytes``) stored in
      the database will survive normal get/put operations, but a bundle
      export → import cycle serializes data through JSON. Because JSON has no
      native binary type, byte values are base64-encoded on export and decoded
      as ``str`` on import. After a roundtrip the value will come back as a
      Python ``str`` instead of ``bytes``.
"""

from contextlib import contextmanager
from datetime import datetime

from ._stratadb import Strata as _Strata

# Exception hierarchy
from ._stratadb import (
    StrataError,
    NotFoundError,
    ValidationError,
    ConflictError,
    StateError,
    ConstraintError,
    AccessDeniedError,
    IoError,
)


# =============================================================================
# Namespace classes
# =============================================================================


class KVNamespace:
    """Namespace for KV Store operations: ``db.kv.put()``, ``db.kv.get()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def put(self, key, value):
        return self._db.kv_put(key, value)

    def get(self, key, *, default=None, as_of=None):
        result = self._db.kv_get(key, as_of=as_of)
        return default if result is None else result

    def delete(self, key):
        return self._db.kv_delete(key)

    def get_versioned(self, key):
        return self._db.kv_get_versioned(key)

    def history(self, key):
        return self._db.kv_history(key)

    def keys(self, *, prefix=None):
        return self._db.kv_list(prefix)

    def list(self, *, prefix=None, limit=None, as_of=None):
        if limit is not None:
            return self._db.kv_list_paginated(prefix=prefix, limit=limit, as_of=as_of)["keys"]
        return self._db.kv_list(prefix, as_of=as_of)


class StateNamespace:
    """Namespace for State Cell operations: ``db.state.set()``, ``db.state.get()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def set(self, cell, value):
        return self._db.state_set(cell, value)

    def get(self, cell, *, default=None, as_of=None):
        result = self._db.state_get(cell, as_of=as_of)
        return default if result is None else result

    def get_versioned(self, cell):
        return self._db.state_get_versioned(cell)

    def init(self, cell, value):
        return self._db.state_init(cell, value)

    def cas(self, cell, new_value, *, expected_version=None):
        return self._db.state_cas(cell, new_value, expected_version)

    def delete(self, cell):
        return self._db.state_delete(cell)

    def list(self, *, prefix=None, as_of=None):
        return self._db.state_list(prefix, as_of=as_of)

    def history(self, cell):
        return self._db.state_history(cell)


class EventsNamespace:
    """Namespace for Event Log operations: ``db.events.append()``, ``db.events.list()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def append(self, event_type, payload):
        return self._db.event_append(event_type, payload)

    def get(self, sequence, *, as_of=None):
        return self._db.event_get(sequence, as_of=as_of)

    def list(self, event_type, *, limit=None, after=None, as_of=None):
        if limit is not None or after is not None:
            return self._db.event_list_paginated(event_type, limit=limit, after=after, as_of=as_of)
        return self._db.event_list(event_type, as_of=as_of)

    @property
    def count(self):
        return self._db.event_len()

    def __len__(self):
        return self._db.event_len()


class JSONNamespace:
    """Namespace for JSON Store operations: ``db.json.set()``, ``db.json.get()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def set(self, key, path, value):
        return self._db.json_set(key, path, value)

    def get(self, key, path="$", *, as_of=None):
        return self._db.json_get(key, path, as_of=as_of)

    def delete(self, key, path="$"):
        return self._db.json_delete(key, path)

    def get_versioned(self, key):
        return self._db.json_get_versioned(key)

    def history(self, key):
        return self._db.json_history(key)

    def list(self, *, prefix=None, limit=100, cursor=None, as_of=None):
        return self._db.json_list(limit, prefix=prefix, cursor=cursor, as_of=as_of)


class Collection:
    """Handle for a single vector collection.

    Returned by ``db.vectors.create()`` or ``db.vectors.collection()``.
    """

    __slots__ = ("_db", "_name")

    def __init__(self, db, name):
        self._db = db
        self._name = name

    @property
    def name(self):
        return self._name

    def upsert(self, key, vector, *, metadata=None):
        return self._db.vector_upsert(self._name, key, vector, metadata)

    def get(self, key, *, as_of=None):
        return self._db.vector_get(self._name, key, as_of=as_of)

    def delete(self, key):
        return self._db.vector_delete(self._name, key)

    def search(self, query, *, k=10, filter=None, metric=None, as_of=None):
        if filter is not None or metric is not None:
            return self._db.vector_search_filtered(
                self._name, query, k, metric=metric, filter=filter, as_of=as_of,
            )
        return self._db.vector_search(self._name, query, k, as_of=as_of)

    def batch_upsert(self, vectors):
        return self._db.vector_batch_upsert(self._name, vectors)

    def stats(self):
        return self._db.vector_collection_stats(self._name)

    def __len__(self):
        return self.stats()["count"]


class VectorsNamespace:
    """Namespace for Vector Store operations: ``db.vectors.create()``, ``db.vectors.collection()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def create(self, name, *, dimension, metric="cosine"):
        self._db.vector_create_collection(name, dimension, metric=metric)
        return Collection(self._db, name)

    def collection(self, name):
        return Collection(self._db, name)

    def drop(self, name):
        return self._db.vector_delete_collection(name)

    def list(self):
        return self._db.vector_list_collections()

    def __contains__(self, name):
        return any(c["name"] == name for c in self._db.vector_list_collections())


class BranchesNamespace:
    """Namespace for branch management: ``db.branches.create()``, ``db.branches.list()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def create(self, name):
        self._db.create_branch(name)

    def delete(self, name):
        self._db.delete_branch(name)

    def exists(self, name):
        return self._db.branch_exists(name)

    def get(self, name):
        return self._db.branch_get(name)

    def list(self):
        return self._db.list_branches()

    def export_bundle(self, branch, path):
        return self._db.branch_export(branch, path)

    def import_bundle(self, path):
        return self._db.branch_import(path)

    def validate_bundle(self, path):
        return self._db.branch_validate_bundle(path)

    def __contains__(self, name):
        return self._db.branch_exists(name)

    def __iter__(self):
        return iter(self._db.list_branches())


class SpacesNamespace:
    """Namespace for space management: ``db.spaces.create()``, ``db.spaces.list()``, etc."""

    __slots__ = ("_db",)

    def __init__(self, db):
        self._db = db

    def create(self, name):
        self._db.space_create(name)

    def delete(self, name, *, force=False):
        if force:
            self._db.delete_space_force(name)
        else:
            self._db.delete_space(name)

    def exists(self, name):
        return self._db.space_exists(name)

    def list(self):
        return self._db.list_spaces()

    def __contains__(self, name):
        return self._db.space_exists(name)

    def __iter__(self):
        return iter(self._db.list_spaces())


# =============================================================================
# Snapshot (time-travel) classes
# =============================================================================


class KVSnapshotNamespace:
    """Read-only KV namespace for time-travel snapshots."""

    __slots__ = ("_db", "_ts")

    def __init__(self, db, ts):
        self._db = db
        self._ts = ts

    def get(self, key, *, default=None):
        result = self._db.kv_get(key, as_of=self._ts)
        return default if result is None else result

    def keys(self, *, prefix=None):
        return self._db.kv_list(prefix, as_of=self._ts)

    def list(self, *, prefix=None, limit=None):
        if limit is not None:
            return self._db.kv_list_paginated(prefix=prefix, limit=limit, as_of=self._ts)["keys"]
        return self._db.kv_list(prefix, as_of=self._ts)


class StateSnapshotNamespace:
    """Read-only State namespace for time-travel snapshots."""

    __slots__ = ("_db", "_ts")

    def __init__(self, db, ts):
        self._db = db
        self._ts = ts

    def get(self, cell, *, default=None):
        result = self._db.state_get(cell, as_of=self._ts)
        return default if result is None else result

    def list(self, *, prefix=None):
        return self._db.state_list(prefix, as_of=self._ts)


class EventsSnapshotNamespace:
    """Read-only Events namespace for time-travel snapshots."""

    __slots__ = ("_db", "_ts")

    def __init__(self, db, ts):
        self._db = db
        self._ts = ts

    def get(self, sequence):
        return self._db.event_get(sequence, as_of=self._ts)

    def list(self, event_type, *, limit=None, after=None):
        if limit is not None or after is not None:
            return self._db.event_list_paginated(event_type, limit=limit, after=after, as_of=self._ts)
        return self._db.event_list(event_type, as_of=self._ts)


class JSONSnapshotNamespace:
    """Read-only JSON namespace for time-travel snapshots."""

    __slots__ = ("_db", "_ts")

    def __init__(self, db, ts):
        self._db = db
        self._ts = ts

    def get(self, key, path="$"):
        return self._db.json_get(key, path, as_of=self._ts)

    def list(self, *, prefix=None, limit=100, cursor=None):
        return self._db.json_list(limit, prefix=prefix, cursor=cursor, as_of=self._ts)


class CollectionSnapshot:
    """Read-only collection handle for time-travel snapshots."""

    __slots__ = ("_db", "_name", "_ts")

    def __init__(self, db, name, ts):
        self._db = db
        self._name = name
        self._ts = ts

    def get(self, key):
        return self._db.vector_get(self._name, key, as_of=self._ts)

    def search(self, query, *, k=10, filter=None, metric=None):
        if filter is not None or metric is not None:
            return self._db.vector_search_filtered(
                self._name, query, k, metric=metric, filter=filter, as_of=self._ts,
            )
        return self._db.vector_search(self._name, query, k, as_of=self._ts)


class VectorsSnapshotNamespace:
    """Read-only Vectors namespace for time-travel snapshots."""

    __slots__ = ("_db", "_ts")

    def __init__(self, db, ts):
        self._db = db
        self._ts = ts

    def collection(self, name):
        return CollectionSnapshot(self._db, name, self._ts)


class Snapshot:
    """Read-only view of the database at a specific point in time.

    Created via ``db.at(timestamp)``. All reads on this object are
    time-travel reads — no ``as_of=`` parameter needed.
    Writes raise ``AttributeError`` (read-only namespaces).
    """

    __slots__ = ("_db", "_ts", "_kv", "_state", "_events", "_json", "_vectors")

    def __init__(self, db, timestamp_us):
        self._db = db
        self._ts = timestamp_us
        self._kv = None
        self._state = None
        self._events = None
        self._json = None
        self._vectors = None

    @property
    def timestamp(self):
        return self._ts

    @property
    def kv(self):
        if self._kv is None:
            self._kv = KVSnapshotNamespace(self._db, self._ts)
        return self._kv

    @property
    def state(self):
        if self._state is None:
            self._state = StateSnapshotNamespace(self._db, self._ts)
        return self._state

    @property
    def events(self):
        if self._events is None:
            self._events = EventsSnapshotNamespace(self._db, self._ts)
        return self._events

    @property
    def json(self):
        if self._json is None:
            self._json = JSONSnapshotNamespace(self._db, self._ts)
        return self._json

    @property
    def vectors(self):
        if self._vectors is None:
            self._vectors = VectorsSnapshotNamespace(self._db, self._ts)
        return self._vectors


# =============================================================================
# Transaction context manager
# =============================================================================


class Transaction:
    """Context manager for database transactions.

    Automatically commits on success and rolls back on exception.

    Usage::

        with db.transaction() as txn:
            db.kv.put("key1", "value1")
            db.kv.put("key2", "value2")
        # Auto-commit on exit

        with db.transaction(read_only=True):
            value = db.kv.get("key1")
        # Read-only transaction
    """

    def __init__(self, db, read_only=False):
        self._db = db
        self._read_only = read_only

    def __enter__(self):
        self._db.begin(self._read_only)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._db.commit()
        else:
            self._db.rollback()
        return False


# =============================================================================
# Main Strata class (composition wrapper)
# =============================================================================


def setup():
    """Download model files for auto-embedding.

    Downloads MiniLM-L6-v2 model files (~80MB) to ~/.stratadb/models/minilm-l6-v2/.
    Called automatically when ``auto_embed=True`` is passed to ``Strata.open()``,
    but can be called explicitly to pre-download during installation.

    Returns:
        The path where model files are stored.

    Raises:
        RuntimeError: If the download fails.

    Example::

        import stratadb
        stratadb.setup()  # pre-download model files
    """
    return _Strata.setup()


class Strata:
    """StrataDB database handle with Pythonic namespace API.

    Wraps the native ``_Strata`` class with namespace accessors
    (``db.kv``, ``db.state``, ``db.events``, ``db.json``, ``db.vectors``,
    ``db.branches``, ``db.spaces``), properties, context managers, and
    time-travel support.
    """

    def __init__(self, _inner):
        # _inner is a native _Strata instance
        object.__setattr__(self, "_inner", _inner)

    _NAMESPACE_HINTS = {
        "kv_": "db.kv",
        "state_": "db.state",
        "event_": "db.events",
        "json_": "db.json",
        "vector_": "db.vectors",
    }

    def __getattr__(self, name):
        for prefix, ns in self._NAMESPACE_HINTS.items():
            if name.startswith(prefix):
                raise AttributeError(
                    f"Strata.{name}() has been removed. "
                    f"Use the namespace API instead: {ns}"
                )
        return getattr(self._inner, name)

    # -- Construction (static) ------------------------------------------------

    @staticmethod
    def open(path, auto_embed=False, read_only=False, embed_batch_size=None):
        """Open a database at the given path."""
        return Strata(_Strata.open(path, auto_embed=auto_embed, read_only=read_only, embed_batch_size=embed_batch_size))

    @staticmethod
    def cache():
        """Create an in-memory database (no persistence)."""
        return Strata(_Strata.cache())

    @staticmethod
    def setup():
        """Download model files for auto-embedding."""
        return _Strata.setup()

    # -- Configuration --------------------------------------------------------

    def config(self):
        """Get the current database configuration.

        Returns a dict with ``"durability"``, ``"auto_embed"``, and
        ``"model"`` (a dict with ``"endpoint"``, ``"model"``, ``"api_key"``,
        ``"timeout_ms"``, or ``None``).
        """
        return self._inner.config()

    def embed_status(self):
        """Return embedding pipeline status.

        Returns a dict with:
        - ``auto_embed``: whether auto-embedding is enabled
        - ``batch_size``: configured embedding batch size
        - ``pending``: items currently in the buffer
        - ``total_queued``: cumulative items pushed
        - ``total_embedded``: cumulative items successfully embedded
        - ``total_failed``: cumulative items that failed
        - ``scheduler_queue_depth``: tasks waiting in scheduler
        - ``scheduler_active_tasks``: tasks currently running
        """
        return self._inner.embed_status()

    @property
    def auto_embed_enabled(self):
        """Whether automatic text embedding is currently enabled (read-only)."""
        return self._inner.auto_embed_enabled()

    def set_auto_embed(self, enabled):
        """Enable or disable automatic text embedding.

        Persisted to ``strata.toml`` for disk-backed databases.

        Args:
            enabled: Whether to enable auto-embed.
        """
        self._inner.set_auto_embed(enabled)

    def configure_model(self, endpoint, model, api_key=None, timeout_ms=None):
        """Configure an inference model endpoint for intelligent search.

        When a model is configured, ``search()`` transparently expands queries
        using the model for better recall.  Without a model, search works the
        same as before.  Persisted to ``strata.toml``.

        Args:
            endpoint: OpenAI-compatible API endpoint URL (e.g. ``"http://localhost:11434/v1"``).
            model: Model name (e.g. ``"qwen3:1.7b"``).
            api_key: Optional bearer token.
            timeout_ms: Request timeout in milliseconds (default 5000).
        """
        self._inner.configure_model(endpoint, model, api_key=api_key, timeout_ms=timeout_ms)

    # -- Search ---------------------------------------------------------------

    def search(self, query, *, k=None, primitives=None, time_range=None,
               mode=None, expand=None, rerank=None):
        """Search across multiple primitives for matching content.

        Args:
            query: Natural-language or keyword query string.
            k: Number of results to return (default: 10).
            primitives: Restrict to specific primitives (e.g. ``["kv", "json"]``).
            time_range: Time range filter as ``{"start": str, "end": str}`` (ISO 8601).
            mode: Search mode: ``"keyword"`` or ``"hybrid"`` (default: ``"hybrid"``).
            expand: Enable/disable query expansion (default: auto).
            rerank: Enable/disable reranking (default: auto).

        Returns:
            List of ``{"entity", "primitive", "score", "rank", "snippet"}`` dicts.
        """
        return self._inner.search(
            query, k=k, primitives=primitives, time_range=time_range,
            mode=mode, expand=expand, rerank=rerank,
        )

    # -- Properties -----------------------------------------------------------

    @property
    def branch(self):
        """Current branch name (read-only)."""
        return self._inner.current_branch()

    @property
    def space(self):
        """Current space name (read-only)."""
        return self._inner.current_space()

    @property
    def in_transaction(self):
        """Whether a transaction is currently active."""
        return self._inner.txn_is_active()

    @property
    def time_range(self):
        """Time range of data in the current branch.

        Returns ``{"oldest_ts": ..., "latest_ts": ...}`` or a dict with
        ``None`` values if the branch has no data.
        """
        return self._inner.time_range()

    # -- Namespace accessors (lazy-cached) ------------------------------------

    @property
    def kv(self):
        """KV Store namespace."""
        try:
            return self.__dict__["_kv"]
        except KeyError:
            ns = KVNamespace(self._inner)
            self.__dict__["_kv"] = ns
            return ns

    @property
    def state(self):
        """State Cell namespace."""
        try:
            return self.__dict__["_state"]
        except KeyError:
            ns = StateNamespace(self._inner)
            self.__dict__["_state"] = ns
            return ns

    @property
    def events(self):
        """Event Log namespace."""
        try:
            return self.__dict__["_events"]
        except KeyError:
            ns = EventsNamespace(self._inner)
            self.__dict__["_events"] = ns
            return ns

    @property
    def json(self):
        """JSON Store namespace."""
        try:
            return self.__dict__["_json"]
        except KeyError:
            ns = JSONNamespace(self._inner)
            self.__dict__["_json"] = ns
            return ns

    @property
    def vectors(self):
        """Vector Store namespace."""
        try:
            return self.__dict__["_vectors"]
        except KeyError:
            ns = VectorsNamespace(self._inner)
            self.__dict__["_vectors"] = ns
            return ns

    @property
    def branches(self):
        """Branch management namespace."""
        try:
            return self.__dict__["_branches"]
        except KeyError:
            ns = BranchesNamespace(self._inner)
            self.__dict__["_branches"] = ns
            return ns

    @property
    def spaces(self):
        """Space management namespace."""
        try:
            return self.__dict__["_spaces"]
        except KeyError:
            ns = SpacesNamespace(self._inner)
            self.__dict__["_spaces"] = ns
            return ns

    # -- Branch operations ----------------------------------------------------

    def checkout(self, name):
        """Switch to a different branch."""
        self._inner.set_branch(name)

    def fork(self, destination):
        """Fork the current branch, copying all data."""
        return self._inner.fork_branch(destination)

    def merge(self, source, *, strategy="last_writer_wins"):
        """Merge a branch into the current branch."""
        return self._inner.merge_branches(source, strategy=strategy)

    def diff(self, branch_a, branch_b):
        """Compare two branches."""
        return self._inner.diff_branches(branch_a, branch_b)

    # -- Space operations -----------------------------------------------------

    def use_space(self, name):
        """Switch to a different space."""
        self._inner.set_space(name)

    # -- Context managers -----------------------------------------------------

    def transaction(self, read_only=False):
        """Start a transaction as a context manager.

        Usage::

            with db.transaction():
                db.kv.put("key1", "value1")
                db.kv.put("key2", "value2")
            # Auto-commit on exit
        """
        return Transaction(self._inner, read_only)

    @contextmanager
    def on_branch(self, name):
        """Temporarily switch to a branch, restoring the original on exit.

        Usage::

            with db.on_branch("experiment"):
                db.kv.put("temp", "value")
            # Back on original branch
        """
        previous = self._inner.current_branch()
        self._inner.set_branch(name)
        try:
            yield
        finally:
            self._inner.set_branch(previous)

    @contextmanager
    def in_space(self, name):
        """Temporarily switch to a space, restoring the original on exit.

        Usage::

            with db.in_space("tenant_42"):
                db.kv.put("key", "value")
            # Back in original space
        """
        previous = self._inner.current_space()
        self._inner.set_space(name)
        try:
            yield
        finally:
            self._inner.set_space(previous)

    # -- Time-travel ----------------------------------------------------------

    def at(self, timestamp):
        """Create a read-only snapshot at a point in time.

        Args:
            timestamp: A ``datetime`` object or an ``int`` (microseconds since epoch).

        Returns:
            A ``Snapshot`` with read-only namespace accessors.
        """
        if isinstance(timestamp, datetime):
            ts = int(timestamp.timestamp() * 1_000_000)
        else:
            ts = int(timestamp)
        return Snapshot(self._inner, ts)


__all__ = [
    "Strata",
    "Transaction",
    "Collection",
    "Snapshot",
    "setup",
    # Exception hierarchy
    "StrataError",
    "NotFoundError",
    "ValidationError",
    "ConflictError",
    "StateError",
    "ConstraintError",
    "AccessDeniedError",
    "IoError",
]
__version__ = "0.13.2"
