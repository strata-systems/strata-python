"""Type stubs for the native _stratadb extension module (PyO3)."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# =============================================================================
# Exception hierarchy
# =============================================================================

class StrataError(Exception):
    """Base exception for all StrataDB errors."""
    ...

class NotFoundError(StrataError):
    """Entity not found."""
    ...

class ValidationError(StrataError):
    """Invalid input or type mismatch."""
    ...

class ConflictError(StrataError):
    """Version or concurrency conflict."""
    ...

class StateError(StrataError):
    """Invalid state transition."""
    ...

class ConstraintError(StrataError):
    """Constraint or limit violation."""
    ...

class AccessDeniedError(StrataError):
    """Access denied (read-only)."""
    ...

class IoError(StrataError):
    """I/O, serialization, or internal error."""
    ...

# =============================================================================
# Value type alias
#
# StrataDB values map to Python as follows:
#   Null   → None
#   Bool   → bool
#   Int    → int
#   Float  → float
#   String → str
#   Bytes  → bytes
#   Array  → list
#   Object → dict
# =============================================================================

Value = None | bool | int | float | str | bytes | list[Any] | dict[str, Any]

# =============================================================================
# Main database class
# =============================================================================

class Strata:
    """StrataDB database handle.

    All operations go through this class. Construct via the ``open`` or
    ``cache`` static methods.
    """

    # -- Construction ---------------------------------------------------------

    @staticmethod
    def open(
        path: str,
        auto_embed: bool = False,
        read_only: bool = False,
        embed_batch_size: Optional[int] = None,
    ) -> "Strata":
        """Open a database at the given path.

        Args:
            path: Directory path for the database.
            auto_embed: Enable automatic text embedding for semantic search.
            read_only: Open in read-only mode.
            embed_batch_size: Number of texts to batch for embedding (default 64).
        """
        ...

    @staticmethod
    def cache() -> "Strata":
        """Create an in-memory database (no persistence)."""
        ...

    @staticmethod
    def setup() -> str:
        """Download model files for auto-embedding.

        Returns:
            The path where model files are stored.

        Raises:
            RuntimeError: If the download fails or the ``embed`` feature is
                not enabled.
        """
        ...

    # -- Configuration --------------------------------------------------------

    def config(self) -> dict[str, Any]:
        """Get the current database configuration.

        Returns a dict with ``"durability"``, ``"auto_embed"``, and
        ``"model"`` (a dict or ``None``).
        """
        ...

    def embed_status(self) -> dict[str, Any]:
        """Get embedding pipeline status.

        Returns a dict with ``auto_embed``, ``batch_size``, ``pending``,
        ``total_queued``, ``total_embedded``, ``total_failed``,
        ``scheduler_queue_depth``, ``scheduler_active_tasks``, ``is_idle``.
        """
        ...

    def auto_embed_enabled(self) -> bool:
        """Check whether auto-embedding is enabled."""
        ...

    def set_auto_embed(self, enabled: bool) -> None:
        """Enable or disable auto-embedding. Persisted to ``strata.toml``."""
        ...

    def configure_model(
        self,
        endpoint: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_ms: Optional[int] = None,
    ) -> None:
        """Configure an inference model endpoint for intelligent search.

        Persisted to ``strata.toml``.

        Args:
            endpoint: OpenAI-compatible API endpoint URL.
            model: Model name (e.g. ``"qwen3:1.7b"``).
            api_key: Optional bearer token for the endpoint.
            timeout_ms: Request timeout in milliseconds (default 5000).
        """
        ...

    def configure_set(self, key: str, value: str) -> None:
        """Set a database configuration key."""
        ...

    def configure_get(self, key: str) -> Optional[str]:
        """Get a database configuration value by key."""
        ...

    # -- KV Store -------------------------------------------------------------

    def kv_put(self, key: str, value: Value) -> int:
        """Store a key-value pair. Returns the version number."""
        ...

    def kv_get(self, key: str, as_of: Optional[int] = None) -> Value | None:
        """Get a value by key. Returns ``None`` if not found.

        Args:
            key: The key to look up.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def kv_delete(self, key: str) -> bool:
        """Delete a key. Returns ``True`` if the key existed."""
        ...

    def kv_list(self, prefix: Optional[str] = None, as_of: Optional[int] = None) -> list[str]:
        """List keys with optional prefix filter.

        Args:
            prefix: Optional prefix to filter keys.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def kv_history(self, key: str) -> list[dict[str, Any]] | None:
        """Get version history for a key.

        Returns a list of ``{"value", "version", "timestamp"}`` dicts,
        or ``None`` if the key does not exist.
        """
        ...

    def kv_get_versioned(self, key: str) -> dict[str, Any] | None:
        """Get a value with version info.

        Returns ``{"value", "version", "timestamp"}`` or ``None``.
        """
        ...

    def kv_batch_put(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch put multiple key-value pairs in a single transaction.

        Each entry dict must have ``"key"`` and ``"value"`` keys.
        Returns a list of ``{"version", "error"}`` dicts (positionally mapped).
        """
        ...

    def kv_list_paginated(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        as_of: Optional[int] = None,
    ) -> dict[str, Any]:
        """List keys with pagination support.

        Returns ``{"keys": [...]}``.

        Note:
            Cursor-based pagination is not yet supported for KV lists.

        Args:
            prefix: Optional prefix to filter keys.
            limit: Optional maximum number of keys to return.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    # -- State Cell -----------------------------------------------------------

    def state_set(self, cell: str, value: Value) -> int:
        """Set a state cell value. Returns the version number."""
        ...

    def state_get(self, cell: str, as_of: Optional[int] = None) -> Value | None:
        """Get a state cell value. Returns ``None`` if not found.

        Args:
            cell: The state cell name.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def state_init(self, cell: str, value: Value) -> int:
        """Initialize a state cell if it doesn't already exist."""
        ...

    def state_cas(
        self,
        cell: str,
        new_value: Value,
        expected_version: Optional[int] = None,
    ) -> int | None:
        """Compare-and-swap update based on version.

        Returns the new version on success, ``None`` on CAS failure.
        """
        ...

    def state_history(self, cell: str) -> list[dict[str, Any]] | None:
        """Get version history for a state cell."""
        ...

    def state_delete(self, cell: str) -> bool:
        """Delete a state cell. Returns ``True`` if it existed."""
        ...

    def state_list(self, prefix: Optional[str] = None, as_of: Optional[int] = None) -> list[str]:
        """List state cell names with optional prefix filter.

        Args:
            prefix: Optional prefix to filter cell names.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def state_get_versioned(self, cell: str) -> dict[str, Any] | None:
        """Get a state cell value with version info."""
        ...

    def state_batch_set(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch set multiple state cells in a single transaction.

        Each entry dict must have ``"cell"`` and ``"value"`` keys.
        Returns a list of ``{"version", "error"}`` dicts (positionally mapped).
        """
        ...

    # -- Event Log ------------------------------------------------------------

    def event_append(self, event_type: str, payload: Value) -> int:
        """Append an event to the log. Returns the sequence number."""
        ...

    def event_get(self, sequence: int, as_of: Optional[int] = None) -> dict[str, Any] | None:
        """Get an event by sequence number.

        Args:
            sequence: The event sequence number.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def event_list(self, event_type: str, as_of: Optional[int] = None) -> list[dict[str, Any]]:
        """List events by type.

        Args:
            event_type: The event type to filter by.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def event_len(self) -> int:
        """Get total event count."""
        ...

    def event_list_paginated(
        self,
        event_type: str,
        limit: Optional[int] = None,
        after: Optional[int] = None,
        as_of: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """List events by type with pagination support.

        Args:
            event_type: The event type to filter by.
            limit: Optional maximum number of events to return.
            after: Optional sequence number to paginate from.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def event_batch_append(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch append multiple events in a single transaction.

        Each entry dict must have ``"event_type"`` and ``"payload"`` keys.
        Returns a list of ``{"version", "error"}`` dicts (positionally mapped).
        """
        ...

    # -- JSON Store -----------------------------------------------------------

    def json_set(self, key: str, path: str, value: Value) -> int:
        """Set a value at a JSONPath. Returns the version number."""
        ...

    def json_get(self, key: str, path: str, as_of: Optional[int] = None) -> Value | None:
        """Get a value at a JSONPath.

        Args:
            key: The document key.
            path: JSONPath expression.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def json_delete(self, key: str, path: str) -> int:
        """Delete a JSON document. Returns the version number."""
        ...

    def json_history(self, key: str) -> list[dict[str, Any]] | None:
        """Get version history for a JSON document."""
        ...

    def json_list(
        self,
        limit: int,
        prefix: Optional[str] = None,
        cursor: Optional[str] = None,
        as_of: Optional[int] = None,
    ) -> dict[str, Any]:
        """List JSON document keys.

        Returns ``{"keys": [...], "cursor": ...}`` where cursor is present
        only when more pages are available.

        Args:
            limit: Maximum number of keys to return.
            prefix: Optional prefix to filter keys.
            cursor: Optional cursor for pagination.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def json_get_versioned(self, key: str) -> dict[str, Any] | None:
        """Get a JSON document value with version info."""
        ...

    def json_batch_set(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Batch set multiple JSON document paths in a single transaction.

        Each entry dict must have ``"key"``, ``"path"``, and ``"value"`` keys.
        Returns a list of ``{"version", "error"}`` dicts (positionally mapped).
        """
        ...

    # -- Vector Store ---------------------------------------------------------

    def vector_create_collection(
        self,
        collection: str,
        dimension: int,
        metric: Optional[str] = None,
    ) -> int:
        """Create a vector collection.

        Args:
            collection: Collection name.
            dimension: Vector dimensionality.
            metric: Distance metric (``"cosine"``, ``"euclidean"``, or
                ``"dot_product"``). Defaults to ``"cosine"``.
        """
        ...

    def vector_delete_collection(self, collection: str) -> bool:
        """Delete a vector collection."""
        ...

    def vector_list_collections(self) -> list[dict[str, Any]]:
        """List vector collections.

        Each dict contains ``name``, ``dimension``, ``metric``, ``count``,
        ``index_type``, ``memory_bytes``.
        """
        ...

    def vector_upsert(
        self,
        collection: str,
        key: str,
        vector: npt.NDArray[np.float32] | list[float],
        metadata: Value | None = None,
    ) -> int:
        """Insert or update a vector. Returns the version number."""
        ...

    def vector_get(self, collection: str, key: str, as_of: Optional[int] = None) -> dict[str, Any] | None:
        """Get a vector by key.

        Returns ``{"key", "embedding", "metadata", "version", "timestamp"}``
        or ``None``.

        Args:
            collection: The collection name.
            key: The vector key.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def vector_delete(self, collection: str, key: str) -> bool:
        """Delete a vector."""
        ...

    def vector_search(
        self,
        collection: str,
        query: npt.NDArray[np.float32] | list[float],
        k: int,
        as_of: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors.

        Returns a list of ``{"key", "score", "metadata"}`` dicts.

        Args:
            collection: The collection name.
            query: Query vector (numpy array or list of floats).
            k: Number of nearest neighbors to return.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    def vector_collection_stats(self, collection: str) -> dict[str, Any]:
        """Get statistics for a single collection."""
        ...

    def vector_batch_upsert(
        self,
        collection: str,
        vectors: list[dict[str, Any]],
    ) -> list[int]:
        """Batch insert/update multiple vectors.

        Each vector dict must have ``"key"`` and ``"vector"`` keys, with
        optional ``"metadata"``.
        """
        ...

    def vector_search_filtered(
        self,
        collection: str,
        query: npt.NDArray[np.float32] | list[float],
        k: int,
        metric: Optional[str] = None,
        filter: Optional[list[dict[str, Any]]] = None,
        as_of: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search for similar vectors with optional filter and metric override.

        Args:
            collection: The collection name.
            query: Query vector (numpy array or list of floats).
            k: Number of nearest neighbors to return.
            metric: Optional distance metric override.
            filter: Optional metadata filters.
            as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
        """
        ...

    # -- Branch Management ----------------------------------------------------

    def current_branch(self) -> str:
        """Get the current branch name."""
        ...

    def set_branch(self, branch: str) -> None:
        """Switch to a different branch."""
        ...

    def create_branch(self, branch: str) -> None:
        """Create a new empty branch."""
        ...

    def fork_branch(self, destination: str) -> dict[str, Any]:
        """Fork the current branch, copying all data.

        Returns ``{"source", "destination", "keys_copied"}``.
        """
        ...

    def list_branches(self) -> list[str]:
        """List all branches."""
        ...

    def delete_branch(self, branch: str) -> None:
        """Delete a branch."""
        ...

    def branch_exists(self, name: str) -> bool:
        """Check if a branch exists."""
        ...

    def branch_get(self, name: str) -> dict[str, Any] | None:
        """Get branch metadata with version info.

        Returns ``{"id", "status", "created_at", "updated_at", "version",
        "timestamp"}`` or ``None``.
        """
        ...

    def diff_branches(
        self, branch_a: str, branch_b: str
    ) -> dict[str, Any]:
        """Compare two branches.

        Returns ``{"branch_a", "branch_b", "summary": {"total_added",
        "total_removed", "total_modified"}}``.
        """
        ...

    def merge_branches(
        self,
        source: str,
        strategy: Optional[str] = None,
    ) -> dict[str, Any]:
        """Merge a branch into the current branch.

        Args:
            source: Branch to merge from.
            strategy: ``"last_writer_wins"`` (default) or ``"strict"``.

        Returns ``{"keys_applied", "spaces_merged", "conflicts"}``.
        """
        ...

    # -- Space Management -----------------------------------------------------

    def current_space(self) -> str:
        """Get the current space name."""
        ...

    def set_space(self, space: str) -> None:
        """Switch to a different space."""
        ...

    def list_spaces(self) -> list[str]:
        """List all spaces in the current branch."""
        ...

    def delete_space(self, space: str) -> None:
        """Delete a space and all its data."""
        ...

    def delete_space_force(self, space: str) -> None:
        """Force delete a space even if non-empty."""
        ...

    def space_create(self, space: str) -> None:
        """Create a new space explicitly."""
        ...

    def space_exists(self, space: str) -> bool:
        """Check if a space exists in the current branch."""
        ...

    # -- Database Operations --------------------------------------------------

    def ping(self) -> str:
        """Check database connectivity."""
        ...

    def info(self) -> dict[str, Any]:
        """Get database info.

        Returns ``{"version", "uptime_secs", "branch_count", "total_keys"}``.
        """
        ...

    def flush(self) -> None:
        """Flush writes to disk."""
        ...

    def compact(self) -> None:
        """Trigger compaction."""
        ...

    # -- Bundle Operations ----------------------------------------------------

    def branch_export(self, branch: str, path: str) -> dict[str, Any]:
        """Export a branch to a bundle file.

        Returns ``{"branch_id", "path", "entry_count", "bundle_size"}``.
        """
        ...

    def branch_import(self, path: str) -> dict[str, Any]:
        """Import a branch from a bundle file.

        Returns ``{"branch_id", "transactions_applied", "keys_written"}``.
        """
        ...

    def branch_validate_bundle(self, path: str) -> dict[str, Any]:
        """Validate a bundle file without importing.

        Returns ``{"branch_id", "format_version", "entry_count",
        "checksums_valid"}``.
        """
        ...

    # -- Transaction Operations -----------------------------------------------

    def begin(self, read_only: Optional[bool] = None) -> None:
        """Begin a new transaction.

        Raises:
            StateError: If a transaction is already active.
        """
        ...

    def commit(self) -> int:
        """Commit the current transaction. Returns the commit version."""
        ...

    def rollback(self) -> None:
        """Rollback the current transaction."""
        ...

    def txn_info(self) -> dict[str, Any] | None:
        """Get current transaction info.

        Returns ``{"id", "status", "started_at"}`` or ``None``.
        """
        ...

    def txn_is_active(self) -> bool:
        """Check if a transaction is currently active."""
        ...

    # -- Search ---------------------------------------------------------------

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        primitives: Optional[list[str]] = None,
        time_range: Optional[dict[str, str]] = None,
        mode: Optional[str] = None,
        expand: Optional[bool] = None,
        rerank: Optional[bool] = None,
    ) -> list[dict[str, Any]]:
        """Search across multiple primitives for matching content.

        Returns a list of ``{"entity", "primitive", "score", "rank",
        "snippet"}`` dicts.
        """
        ...

    # -- Time-Travel ----------------------------------------------------------

    def time_range(self) -> dict[str, Any]:
        """Get the time range of data in the current branch.

        Returns ``{"oldest_ts", "latest_ts"}`` where values are microseconds
        since epoch, or ``None`` if the branch has no data.
        """
        ...

    # -- Embedding ------------------------------------------------------------

    def embed(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a single text string into a vector.

        Returns a numpy array of float32 values.
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[npt.NDArray[np.float32]]:
        """Embed multiple texts into vectors.

        Returns a list of numpy arrays.
        """
        ...

    # -- Inference ------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop_tokens: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Generate text using a loaded model.

        Returns ``{"text", "stop_reason", "prompt_tokens",
        "completion_tokens", "model"}``.
        """
        ...

    def tokenize(
        self,
        model: str,
        text: str,
        add_special_tokens: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Tokenize text into token IDs.

        Returns ``{"ids", "count", "model"}``.
        """
        ...

    def detokenize(self, model: str, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        ...

    def generate_unload(self, model: str) -> bool:
        """Unload a model from memory.

        Returns ``True`` if the model was loaded and is now unloaded.
        """
        ...

    # -- Model Management -----------------------------------------------------

    def models_list(self) -> list[dict[str, Any]]:
        """List available models from the registry.

        Each dict contains ``name``, ``task``, ``architecture``,
        ``default_quant``, ``embedding_dim``, ``is_local``, ``size_bytes``.
        """
        ...

    def models_pull(self, name: str) -> dict[str, Any]:
        """Download a model by name.

        Returns ``{"name", "path"}``.
        """
        ...

    def models_local(self) -> list[dict[str, Any]]:
        """List locally downloaded models.

        Each dict contains ``name``, ``task``, ``architecture``,
        ``default_quant``, ``embedding_dim``, ``is_local``, ``size_bytes``.
        """
        ...

    # -- Durability -----------------------------------------------------------

    def durability_counters(self) -> dict[str, Any]:
        """Get WAL durability counters.

        Returns ``{"wal_appends", "sync_calls", "bytes_written", "sync_nanos"}``.
        """
        ...

    # -- Retention ------------------------------------------------------------

    def retention_apply(self) -> None:
        """Apply retention policy (trigger garbage collection)."""
        ...
