# strata-python

Python SDK for [StrataDB](https://github.com/strata-systems/strata-core) - an embedded database for AI agents.

PyO3-based bindings embedding the Rust library directly in a Python native extension. No network hop, no serialization overhead beyond the Python/Rust boundary.

## Installation

### From PyPI (coming soon)

```bash
pip install stratadb
```

### From Source

Requires Rust toolchain and maturin:

```bash
git clone https://github.com/strata-systems/strata-python.git
cd strata-python
pip install maturin
maturin develop
```

## Quick Start

```python
from stratadb import Strata

# Open a database (or use Strata.cache() for in-memory)
db = Strata.open("/path/to/data")

# Key-value storage
db.kv_put("user:123", "Alice")
print(db.kv_get("user:123"))  # "Alice"

# Branch isolation (like git branches)
db.create_branch("experiment")
db.set_branch("experiment")
print(db.kv_get("user:123"))  # None - isolated

# Space organization within branches
db.set_space("conversations")
db.kv_put("msg_001", "hello")
```

## Features

### Six Data Primitives

| Primitive | Purpose | Key Methods |
|-----------|---------|-------------|
| **KV Store** | Working memory, config | `kv_put`, `kv_get`, `kv_delete`, `kv_list` |
| **Event Log** | Immutable audit trail | `event_append`, `event_get`, `event_list` |
| **State Cell** | CAS-based coordination | `state_set`, `state_get`, `state_cas` |
| **JSON Store** | Structured documents | `json_set`, `json_get`, `json_delete` |
| **Vector Store** | Embeddings, similarity search | `vector_upsert`, `vector_search` |
| **Branch** | Data isolation | `create_branch`, `set_branch`, `fork_branch` |

### NumPy Integration

Vector operations accept and return NumPy arrays:

```python
import numpy as np

# Create a collection
db.vector_create_collection("embeddings", 384, metric="cosine")

# Upsert with NumPy array
embedding = np.random.rand(384).astype(np.float32)
db.vector_upsert("embeddings", "doc-1", embedding, metadata={"title": "Hello"})

# Search returns matches with scores
results = db.vector_search("embeddings", embedding, k=10)
for match in results:
    print(f"{match['key']}: {match['score']}")
```

### Branch Operations

```python
# Fork current branch (copies all data)
db.fork_branch("experiment")

# Compare branches
diff = db.diff_branches("default", "experiment")
print(f"Added: {diff['summary']['total_added']}")

# Merge branches
result = db.merge_branches("experiment", strategy="last_writer_wins")
print(f"Keys applied: {result['keys_applied']}")
```

### Event Log

```python
# Append events
db.event_append("tool_call", {"tool": "search", "query": "weather"})
db.event_append("tool_call", {"tool": "calculator", "expr": "2+2"})

# Get by sequence number
event = db.event_get(0)

# List by type
tool_calls = db.event_list("tool_call")
```

### Compare-and-Swap (Version-based)

```python
# Initialize if not exists
db.state_init("counter", 0)

# CAS is version-based - pass expected version, not expected value
# Get current value and its version via state_set (returns version)
version = db.state_set("counter", 1)

# Update only if version matches
new_version = db.state_cas("counter", 2, version)  # (cell, new_value, expected_version)
if new_version is None:
    print("CAS failed - version mismatch")
```

## API Reference

### Strata

| Method | Description |
|--------|-------------|
| `Strata.open(path)` | Open database at path |
| `Strata.cache()` | Create in-memory database |

### KV Store

| Method | Description |
|--------|-------------|
| `kv_put(key, value)` | Store a value |
| `kv_get(key)` | Get a value (returns None if missing) |
| `kv_delete(key)` | Delete a key |
| `kv_list(prefix=None)` | List keys |
| `kv_history(key)` | Get version history |

### State Cell

| Method | Description |
|--------|-------------|
| `state_set(cell, value)` | Set value |
| `state_get(cell)` | Get value |
| `state_init(cell, value)` | Initialize if not exists |
| `state_cas(cell, new_value, expected_version)` | Compare-and-swap (version-based) |

### Event Log

| Method | Description |
|--------|-------------|
| `event_append(type, payload)` | Append event |
| `event_get(sequence)` | Get by sequence |
| `event_list(type)` | List by type |
| `event_len()` | Get count |

### JSON Store

| Method | Description |
|--------|-------------|
| `json_set(key, path, value)` | Set at JSONPath |
| `json_get(key, path)` | Get at JSONPath |
| `json_delete(key, path)` | Delete |
| `json_list(limit, prefix, cursor)` | List keys |

### Vector Store

| Method | Description |
|--------|-------------|
| `vector_create_collection(name, dim, metric)` | Create collection |
| `vector_delete_collection(name)` | Delete collection |
| `vector_list_collections()` | List collections |
| `vector_upsert(collection, key, vector, metadata)` | Insert/update |
| `vector_get(collection, key)` | Get vector |
| `vector_delete(collection, key)` | Delete vector |
| `vector_search(collection, query, k)` | Search |

### Branches

| Method | Description |
|--------|-------------|
| `current_branch()` | Get current branch |
| `set_branch(name)` | Switch branch |
| `create_branch(name)` | Create empty branch |
| `fork_branch(dest)` | Fork with data copy |
| `list_branches()` | List all branches |
| `delete_branch(name)` | Delete branch |
| `diff_branches(a, b)` | Compare branches |
| `merge_branches(source, strategy)` | Merge into current |

### Spaces

| Method | Description |
|--------|-------------|
| `current_space()` | Get current space |
| `set_space(name)` | Switch space |
| `list_spaces()` | List spaces |
| `delete_space(name)` | Delete space |

### Database

| Method | Description |
|--------|-------------|
| `ping()` | Health check |
| `info()` | Get database info |
| `flush()` | Flush to disk |
| `compact()` | Trigger compaction |

## Development

```bash
# Install dev dependencies
pip install maturin pytest numpy

# Build and install in development mode
maturin develop

# Run tests
pytest tests/
```

## License

MIT
