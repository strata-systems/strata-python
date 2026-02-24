//! Python bindings for StrataDB.
//!
//! This module exposes the StrataDB API to Python via PyO3.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::sync::Mutex;

use ::stratadb::{
    AccessMode, BatchEventEntry, BatchItemResult, BatchJsonEntry, BatchKvEntry, BatchStateEntry,
    BatchVectorEntry, BranchExportResult, BranchImportResult, BundleValidateResult,
    CollectionInfo, Command, DistanceMetric, Error as StrataError, FilterOp, GenerationResult,
    MergeStrategy, MetadataFilter, ModelInfoOutput, OpenOptions, Output, SearchQuery, Session,
    Strata as RustStrata, TimeRangeInput, TokenizeResult, Value, VersionedBranchInfo,
    VersionedValue, WalCounters,
};

// =============================================================================
// Exception hierarchy (#4)
// =============================================================================

pyo3::create_exception!(_stratadb, StrataError_,   pyo3::exceptions::PyException, "Base exception for all StrataDB errors.");
pyo3::create_exception!(_stratadb, NotFoundError,   StrataError_, "Entity not found.");
pyo3::create_exception!(_stratadb, ValidationError, StrataError_, "Invalid input or type mismatch.");
pyo3::create_exception!(_stratadb, ConflictError,   StrataError_, "Version or concurrency conflict.");
pyo3::create_exception!(_stratadb, StateError,      StrataError_, "Invalid state transition.");
pyo3::create_exception!(_stratadb, ConstraintError, StrataError_, "Constraint or limit violation.");
pyo3::create_exception!(_stratadb, AccessDeniedError, StrataError_, "Access denied (read-only).");
pyo3::create_exception!(_stratadb, IoError,         StrataError_, "I/O, serialization, or internal error.");

/// Convert a Python object to a stratadb Value.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        Ok(Value::Null)
    } else if let Ok(b) = obj.extract::<bool>() {
        Ok(Value::Bool(b))
    } else if let Ok(i) = obj.extract::<i64>() {
        Ok(Value::Int(i))
    } else if let Ok(f) = obj.extract::<f64>() {
        Ok(Value::Float(f))
    } else if let Ok(s) = obj.extract::<String>() {
        Ok(Value::String(s))
    } else if let Ok(bytes) = obj.extract::<Vec<u8>>() {
        Ok(Value::Bytes(bytes))
    } else if let Ok(list) = obj.downcast::<PyList>() {
        let values: PyResult<Vec<Value>> = list
            .iter()
            .map(|item| py_to_value(&item))
            .collect();
        Ok(Value::Array(values?))
    } else if let Ok(dict) = obj.downcast::<PyDict>() {
        let mut map = HashMap::new();
        for (key, val) in dict.iter() {
            let key_str: String = key.extract()?;
            map.insert(key_str, py_to_value(&val)?);
        }
        Ok(Value::Object(map))
    } else {
        Err(ValidationError::new_err("Unsupported value type"))
    }
}

/// Convert a stratadb Value to a Python object.
fn value_to_py(py: Python<'_>, value: Value) -> PyResult<PyObject> {
    use pyo3::conversion::ToPyObject;
    match value {
        Value::Null => Ok(py.None()),
        Value::Bool(b) => Ok(b.to_object(py)),
        Value::Int(i) => Ok(i.to_object(py)),
        Value::Float(f) => Ok(f.to_object(py)),
        Value::String(s) => Ok(s.to_object(py)),
        Value::Bytes(b) => Ok(b.to_object(py)),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(value_to_py(py, item)?)?;
            }
            Ok(list.unbind().into_any())
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, value_to_py(py, v)?)?;
            }
            Ok(dict.unbind().into_any())
        }
    }
}

/// Convert a VersionedValue to a Python dict.
fn versioned_to_py(py: Python<'_>, vv: VersionedValue) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("value", value_to_py(py, vv.value)?)?;
    dict.set_item("version", vv.version)?;
    dict.set_item("timestamp", vv.timestamp)?;
    Ok(dict.unbind().into_any())
}

/// Convert stratadb error to PyErr with proper exception type (#4).
fn to_py_err(e: StrataError) -> PyErr {
    let msg = format!("{}", e);
    match e {
        // Not Found
        StrataError::KeyNotFound { .. }
        | StrataError::BranchNotFound { .. }
        | StrataError::CollectionNotFound { .. }
        | StrataError::StreamNotFound { .. }
        | StrataError::CellNotFound { .. }
        | StrataError::DocumentNotFound { .. } => NotFoundError::new_err(msg),

        // Validation
        StrataError::InvalidKey { .. }
        | StrataError::InvalidPath { .. }
        | StrataError::InvalidInput { .. }
        | StrataError::WrongType { .. } => ValidationError::new_err(msg),

        // Conflict
        StrataError::VersionConflict { .. }
        | StrataError::TransitionFailed { .. }
        | StrataError::Conflict { .. }
        | StrataError::TransactionConflict { .. } => ConflictError::new_err(msg),

        // State
        StrataError::BranchClosed { .. }
        | StrataError::BranchExists { .. }
        | StrataError::CollectionExists { .. }
        | StrataError::TransactionNotActive
        | StrataError::TransactionAlreadyActive => StateError::new_err(msg),

        // Constraint
        StrataError::DimensionMismatch { .. }
        | StrataError::ConstraintViolation { .. }
        | StrataError::HistoryTrimmed { .. }
        | StrataError::Overflow { .. } => ConstraintError::new_err(msg),

        // Access
        StrataError::AccessDenied { .. } => AccessDeniedError::new_err(msg),

        // Constraint (history trimmed / unavailable)
        StrataError::HistoryUnavailable { .. } => ConstraintError::new_err(msg),

        // I/O / System
        StrataError::Io { .. }
        | StrataError::Serialization { .. }
        | StrataError::Internal { .. }
        | StrataError::NotImplemented { .. } => IoError::new_err(msg),
    }
}

/// StrataDB database handle.
///
/// This is the main entry point for interacting with StrataDB from Python.
/// All operations go through this class.
#[pyclass(name = "Strata", subclass)]
pub struct PyStrata {
    inner: RustStrata,
    /// Session for transaction support. Lazily initialized.
    session: Mutex<Option<Session>>,
}

#[pymethods]
impl PyStrata {
    /// Open a database at the given path.
    ///
    /// Args:
    ///     path: Directory path for the database.
    ///     auto_embed: Enable automatic text embedding for semantic search.
    ///     read_only: Open in read-only mode.
    ///     embed_batch_size: Number of texts to batch for embedding (default 64).
    ///     embed_model: Embedding model name ("miniLM", "nomic-embed", "bge-m3", "gemma-embed").
    #[staticmethod]
    #[pyo3(signature = (path, auto_embed=false, read_only=false, embed_batch_size=None, embed_model=None))]
    fn open(py: Python<'_>, path: &str, auto_embed: bool, read_only: bool, embed_batch_size: Option<usize>, embed_model: Option<&str>) -> PyResult<Self> {
        // Auto-download model files when auto_embed is requested (best-effort).
        #[cfg(feature = "embed")]
        if auto_embed {
            if let Err(e) = strata_intelligence::embed::download::ensure_model() {
                PyErr::warn_bound(
                    py,
                    &py.get_type_bound::<pyo3::exceptions::PyUserWarning>(),
                    &format!("Failed to download embedding model: {}", e),
                    1,
                )?;
            }
        }

        // Suppress unused variable warning when embed feature is disabled.
        #[cfg(not(feature = "embed"))]
        let _ = py;

        let mut opts = OpenOptions::new();
        if auto_embed {
            opts = opts.auto_embed(true);
        }
        if read_only {
            opts = opts.access_mode(AccessMode::ReadOnly);
        }
        if let Some(bs) = embed_batch_size {
            opts = opts.embed_batch_size(bs);
        }
        if let Some(model) = embed_model {
            opts = opts.embed_model(model);
        }

        let inner = RustStrata::open_with(path, opts).map_err(to_py_err)?;
        Ok(Self {
            inner,
            session: Mutex::new(None),
        })
    }

    /// Create an in-memory database (no persistence).
    #[staticmethod]
    fn cache() -> PyResult<Self> {
        let inner = RustStrata::cache().map_err(to_py_err)?;
        Ok(Self {
            inner,
            session: Mutex::new(None),
        })
    }

    /// Download model files for auto-embedding.
    ///
    /// Downloads MiniLM-L6-v2 model files to ~/.stratadb/models/minilm-l6-v2/.
    /// Called automatically when auto_embed=True, but can be called explicitly
    /// to pre-download (e.g., during pip install).
    ///
    /// Returns the path where model files are stored.
    #[staticmethod]
    fn setup() -> PyResult<String> {
        #[cfg(feature = "embed")]
        {
            let path = strata_intelligence::embed::download::ensure_model()
                .map_err(|e| PyRuntimeError::new_err(e))?;
            Ok(path.to_string_lossy().into_owned())
        }

        #[cfg(not(feature = "embed"))]
        {
            Err(PyRuntimeError::new_err(
                "The 'embed' feature is not enabled in this build",
            ))
        }
    }

    // =========================================================================
    // KV Store
    // =========================================================================

    /// Store a key-value pair.
    fn kv_put(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let v = py_to_value(value)?;
        self.inner.kv_put(key, v).map_err(to_py_err)
    }

    /// Get a value by key.
    ///
    /// Args:
    ///     key: The key to look up.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (key, as_of=None))]
    fn kv_get(&self, py: Python<'_>, key: &str, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::KvGet {
                branch: None, space: None, key: key.to_string(), as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::Maybe(Some(v)) => Ok(value_to_py(py, v)?),
                Output::Maybe(None) => Ok(py.None()),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            match self.inner.kv_get(key).map_err(to_py_err)? {
                Some(v) => Ok(value_to_py(py, v)?),
                None => Ok(py.None()),
            }
        }
    }

    /// Delete a key.
    fn kv_delete(&self, key: &str) -> PyResult<bool> {
        self.inner.kv_delete(key).map_err(to_py_err)
    }

    /// List keys with optional prefix filter.
    ///
    /// Args:
    ///     prefix: Optional prefix to filter keys.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (prefix=None, as_of=None))]
    fn kv_list(&self, prefix: Option<&str>, as_of: Option<u64>) -> PyResult<Vec<String>> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::KvList {
                branch: None, space: None,
                prefix: prefix.map(|s| s.to_string()),
                cursor: None, limit: None,
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::Keys(keys) => Ok(keys),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            self.inner.kv_list(prefix).map_err(to_py_err)
        }
    }

    /// Get version history for a key.
    fn kv_history(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.kv_getv(key).map_err(to_py_err)? {
            Some(versions) => {
                let list = PyList::empty_bound(py);
                for vv in versions {
                    list.append(versioned_to_py(py, vv)?)?;
                }
                Ok(list.unbind().into_any())
            }
            None => Ok(py.None()),
        }
    }

    // =========================================================================
    // State Cell
    // =========================================================================

    /// Set a state cell value.
    fn state_set(&self, cell: &str, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let v = py_to_value(value)?;
        self.inner.state_set(cell, v).map_err(to_py_err)
    }

    /// Get a state cell value.
    ///
    /// Args:
    ///     cell: The state cell name.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (cell, as_of=None))]
    fn state_get(&self, py: Python<'_>, cell: &str, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::StateGet {
                branch: None, space: None, cell: cell.to_string(), as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::Maybe(Some(v)) => Ok(value_to_py(py, v)?),
                Output::Maybe(None) => Ok(py.None()),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            match self.inner.state_get(cell).map_err(to_py_err)? {
                Some(v) => Ok(value_to_py(py, v)?),
                None => Ok(py.None()),
            }
        }
    }

    /// Initialize a state cell if it doesn't exist.
    fn state_init(&self, cell: &str, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let v = py_to_value(value)?;
        self.inner.state_init(cell, v).map_err(to_py_err)
    }

    /// Compare-and-swap update based on version.
    #[pyo3(signature = (cell, new_value, expected_version=None))]
    fn state_cas(
        &self,
        cell: &str,
        new_value: &Bound<'_, PyAny>,
        expected_version: Option<u64>,
    ) -> PyResult<Option<u64>> {
        let new = py_to_value(new_value)?;
        self.inner.state_cas(cell, expected_version, new).map_err(to_py_err)
    }

    /// Get version history for a state cell.
    fn state_history(&self, py: Python<'_>, cell: &str) -> PyResult<PyObject> {
        match self.inner.state_getv(cell).map_err(to_py_err)? {
            Some(versions) => {
                let list = PyList::empty_bound(py);
                for vv in versions {
                    list.append(versioned_to_py(py, vv)?)?;
                }
                Ok(list.unbind().into_any())
            }
            None => Ok(py.None()),
        }
    }

    // =========================================================================
    // Event Log
    // =========================================================================

    /// Append an event to the log.
    fn event_append(&self, event_type: &str, payload: &Bound<'_, PyAny>) -> PyResult<u64> {
        let v = py_to_value(payload)?;
        self.inner.event_append(event_type, v).map_err(to_py_err)
    }

    /// Get an event by sequence number.
    ///
    /// Args:
    ///     sequence: The event sequence number.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (sequence, as_of=None))]
    fn event_get(&self, py: Python<'_>, sequence: u64, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::EventGet {
                branch: None, space: None, sequence, as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::MaybeVersioned(Some(vv)) => Ok(versioned_to_py(py, vv)?),
                Output::MaybeVersioned(None) => Ok(py.None()),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            match self.inner.event_get(sequence).map_err(to_py_err)? {
                Some(vv) => Ok(versioned_to_py(py, vv)?),
                None => Ok(py.None()),
            }
        }
    }

    /// List events by type.
    ///
    /// Args:
    ///     event_type: The event type to filter by.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (event_type, as_of=None))]
    fn event_list(&self, py: Python<'_>, event_type: &str, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::EventGetByType {
                branch: None, space: None,
                event_type: event_type.to_string(),
                limit: None, after_sequence: None,
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::VersionedValues(events) => {
                    let list = PyList::empty_bound(py);
                    for vv in events {
                        list.append(versioned_to_py(py, vv)?)?;
                    }
                    Ok(list.unbind().into_any())
                }
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            let events = self.inner.event_get_by_type(event_type).map_err(to_py_err)?;
            let list = PyList::empty_bound(py);
            for vv in events {
                list.append(versioned_to_py(py, vv)?)?;
            }
            Ok(list.unbind().into_any())
        }
    }

    /// Get total event count.
    fn event_len(&self) -> PyResult<u64> {
        self.inner.event_len().map_err(to_py_err)
    }

    // =========================================================================
    // JSON Store
    // =========================================================================

    /// Set a value at a JSONPath.
    fn json_set(&self, key: &str, path: &str, value: &Bound<'_, PyAny>) -> PyResult<u64> {
        let v = py_to_value(value)?;
        self.inner.json_set(key, path, v).map_err(to_py_err)
    }

    /// Get a value at a JSONPath.
    ///
    /// Args:
    ///     key: The document key.
    ///     path: JSONPath expression.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (key, path, as_of=None))]
    fn json_get(&self, py: Python<'_>, key: &str, path: &str, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::JsonGet {
                branch: None, space: None,
                key: key.to_string(), path: path.to_string(),
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::Maybe(Some(v)) => Ok(value_to_py(py, v)?),
                Output::Maybe(None) => Ok(py.None()),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            match self.inner.json_get(key, path).map_err(to_py_err)? {
                Some(v) => Ok(value_to_py(py, v)?),
                None => Ok(py.None()),
            }
        }
    }

    /// Delete a JSON document.
    fn json_delete(&self, key: &str, path: &str) -> PyResult<u64> {
        self.inner.json_delete(key, path).map_err(to_py_err)
    }

    /// Get version history for a JSON document.
    fn json_history(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.json_getv(key).map_err(to_py_err)? {
            Some(versions) => {
                let list = PyList::empty_bound(py);
                for vv in versions {
                    list.append(versioned_to_py(py, vv)?)?;
                }
                Ok(list.unbind().into_any())
            }
            None => Ok(py.None()),
        }
    }

    /// List JSON document keys.
    ///
    /// Args:
    ///     limit: Maximum number of keys to return.
    ///     prefix: Optional prefix to filter keys.
    ///     cursor: Optional cursor for pagination.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (limit, prefix=None, cursor=None, as_of=None))]
    fn json_list(
        &self,
        py: Python<'_>,
        limit: u64,
        prefix: Option<&str>,
        cursor: Option<&str>,
        as_of: Option<u64>,
    ) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::JsonList {
                branch: None, space: None,
                prefix: prefix.map(|s| s.to_string()),
                cursor: cursor.map(|s| s.to_string()),
                limit,
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::JsonListResult { keys, cursor: next_cursor } => {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("keys", keys)?;
                    if let Some(c) = next_cursor {
                        dict.set_item("cursor", c)?;
                    }
                    Ok(dict.unbind().into_any())
                }
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            let (keys, next_cursor) = self
                .inner
                .json_list(prefix.map(|s| s.to_string()), cursor.map(|s| s.to_string()), limit)
                .map_err(to_py_err)?;
            let dict = PyDict::new_bound(py);
            dict.set_item("keys", keys)?;
            if let Some(c) = next_cursor {
                dict.set_item("cursor", c)?;
            }
            Ok(dict.unbind().into_any())
        }
    }

    // =========================================================================
    // Vector Store
    // =========================================================================

    /// Create a vector collection.
    #[pyo3(signature = (collection, dimension, metric=None))]
    fn vector_create_collection(
        &self,
        collection: &str,
        dimension: u64,
        metric: Option<&str>,
    ) -> PyResult<u64> {
        let m = match metric.unwrap_or("cosine") {
            "cosine" => DistanceMetric::Cosine,
            "euclidean" => DistanceMetric::Euclidean,
            "dot_product" | "dotproduct" => DistanceMetric::DotProduct,
            _ => return Err(ValidationError::new_err("Invalid metric")),
        };
        self.inner
            .vector_create_collection(collection, dimension, m)
            .map_err(to_py_err)
    }

    /// Delete a vector collection.
    fn vector_delete_collection(&self, collection: &str) -> PyResult<bool> {
        self.inner
            .vector_delete_collection(collection)
            .map_err(to_py_err)
    }

    /// List vector collections.
    fn vector_list_collections(&self, py: Python<'_>) -> PyResult<PyObject> {
        let collections = self.inner.vector_list_collections().map_err(to_py_err)?;
        let list = PyList::empty_bound(py);
        for c in collections {
            let dict = collection_info_to_py(py, c)?;
            list.append(dict)?;
        }
        Ok(list.unbind().into_any())
    }

    /// Insert or update a vector.
    #[pyo3(signature = (collection, key, vector, metadata=None))]
    fn vector_upsert(
        &self,
        collection: &str,
        key: &str,
        vector: &Bound<'_, PyAny>,
        metadata: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<u64> {
        let vec = extract_vector(vector)?;
        let meta = match metadata {
            Some(m) => Some(py_to_value(m)?),
            None => None,
        };
        self.inner
            .vector_upsert(collection, key, vec, meta)
            .map_err(to_py_err)
    }

    /// Get a vector by key.
    ///
    /// Args:
    ///     collection: The collection name.
    ///     key: The vector key.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (collection, key, as_of=None))]
    fn vector_get(&self, py: Python<'_>, collection: &str, key: &str, as_of: Option<u64>) -> PyResult<PyObject> {
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::VectorGet {
                branch: None, space: None,
                collection: collection.to_string(),
                key: key.to_string(),
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::VectorData(Some(vd)) => {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("key", &vd.key)?;
                    let arr = PyArray1::from_slice_bound(py, &vd.data.embedding);
                    dict.set_item("embedding", arr)?;
                    if let Some(meta) = vd.data.metadata {
                        dict.set_item("metadata", value_to_py(py, meta)?)?;
                    }
                    dict.set_item("version", vd.version)?;
                    dict.set_item("timestamp", vd.timestamp)?;
                    Ok(dict.unbind().into_any())
                }
                Output::VectorData(None) => Ok(py.None()),
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            match self.inner.vector_get(collection, key).map_err(to_py_err)? {
                Some(vd) => {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("key", &vd.key)?;
                    let arr = PyArray1::from_slice_bound(py, &vd.data.embedding);
                    dict.set_item("embedding", arr)?;
                    if let Some(meta) = vd.data.metadata {
                        dict.set_item("metadata", value_to_py(py, meta)?)?;
                    }
                    dict.set_item("version", vd.version)?;
                    dict.set_item("timestamp", vd.timestamp)?;
                    Ok(dict.unbind().into_any())
                }
                None => Ok(py.None()),
            }
        }
    }

    /// Delete a vector.
    fn vector_delete(&self, collection: &str, key: &str) -> PyResult<bool> {
        self.inner.vector_delete(collection, key).map_err(to_py_err)
    }

    /// Search for similar vectors.
    ///
    /// Args:
    ///     collection: The collection name.
    ///     query: Query vector (numpy array or list of floats).
    ///     k: Number of nearest neighbors to return.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (collection, query, k, as_of=None))]
    fn vector_search(
        &self,
        py: Python<'_>,
        collection: &str,
        query: &Bound<'_, PyAny>,
        k: u64,
        as_of: Option<u64>,
    ) -> PyResult<PyObject> {
        let vec = extract_vector(query)?;
        if let Some(ts) = as_of {
            match self.inner.executor().execute(Command::VectorSearch {
                branch: None, space: None,
                collection: collection.to_string(),
                query: vec, k,
                filter: None, metric: None,
                as_of: Some(ts),
            }).map_err(to_py_err)? {
                Output::VectorMatches(matches) => {
                    let list = PyList::empty_bound(py);
                    for m in matches {
                        let dict = PyDict::new_bound(py);
                        dict.set_item("key", m.key)?;
                        dict.set_item("score", m.score)?;
                        if let Some(meta) = m.metadata {
                            dict.set_item("metadata", value_to_py(py, meta)?)?;
                        }
                        list.append(dict)?;
                    }
                    Ok(list.unbind().into_any())
                }
                _ => Err(PyRuntimeError::new_err("Unexpected output")),
            }
        } else {
            let matches = self
                .inner
                .vector_search(collection, vec, k)
                .map_err(to_py_err)?;
            let list = PyList::empty_bound(py);
            for m in matches {
                let dict = PyDict::new_bound(py);
                dict.set_item("key", m.key)?;
                dict.set_item("score", m.score)?;
                if let Some(meta) = m.metadata {
                    dict.set_item("metadata", value_to_py(py, meta)?)?;
                }
                list.append(dict)?;
            }
            Ok(list.unbind().into_any())
        }
    }

    /// Get statistics for a single collection.
    fn vector_collection_stats(&self, py: Python<'_>, collection: &str) -> PyResult<PyObject> {
        let info = self
            .inner
            .vector_collection_stats(collection)
            .map_err(to_py_err)?;
        collection_info_to_py(py, info)
    }

    /// Batch insert/update multiple vectors.
    ///
    /// Each vector should be a dict with 'key', 'vector', and optional 'metadata'.
    fn vector_batch_upsert(
        &self,
        collection: &str,
        vectors: &Bound<'_, PyList>,
    ) -> PyResult<Vec<u64>> {
        let batch: Vec<BatchVectorEntry> = vectors
            .iter()
            .map(|item| {
                let dict = item.downcast::<PyDict>()?;
                let key: String = dict
                    .get_item("key")?
                    .ok_or_else(|| ValidationError::new_err("missing 'key'"))?
                    .extract()?;
                let vec = extract_vector(
                    &dict
                        .get_item("vector")?
                        .ok_or_else(|| ValidationError::new_err("missing 'vector'"))?,
                )?;
                let meta = dict
                    .get_item("metadata")?
                    .map(|m| py_to_value(&m))
                    .transpose()?;
                Ok(BatchVectorEntry {
                    key,
                    vector: vec,
                    metadata: meta,
                })
            })
            .collect::<PyResult<_>>()?;
        self.inner
            .vector_batch_upsert(collection, batch)
            .map_err(to_py_err)
    }

    // =========================================================================
    // Batch Operations
    // =========================================================================

    /// Batch put multiple key-value pairs in a single transaction.
    ///
    /// Each entry should be a dict with 'key' and 'value'.
    /// Returns a list of dicts with 'version' (on success) or 'error' (on failure).
    fn kv_batch_put(
        &self,
        py: Python<'_>,
        entries: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let batch: Vec<BatchKvEntry> = entries
            .iter()
            .map(|item| {
                let dict = item.downcast::<PyDict>()?;
                let key: String = dict
                    .get_item("key")?
                    .ok_or_else(|| ValidationError::new_err("missing 'key'"))?
                    .extract()?;
                let value = py_to_value(
                    &dict
                        .get_item("value")?
                        .ok_or_else(|| ValidationError::new_err("missing 'value'"))?,
                )?;
                Ok(BatchKvEntry { key, value })
            })
            .collect::<PyResult<_>>()?;

        match self
            .inner
            .executor()
            .execute(Command::KvBatchPut {
                branch: None,
                space: None,
                entries: batch,
            })
            .map_err(to_py_err)?
        {
            Output::BatchResults(results) => batch_results_to_py(py, results),
            _ => Err(PyRuntimeError::new_err("Unexpected output for KvBatchPut")),
        }
    }

    /// Batch append multiple events in a single transaction.
    ///
    /// Each entry should be a dict with 'event_type' and 'payload'.
    /// Returns a list of dicts with 'version' (on success) or 'error' (on failure).
    fn event_batch_append(
        &self,
        py: Python<'_>,
        entries: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let batch: Vec<BatchEventEntry> = entries
            .iter()
            .map(|item| {
                let dict = item.downcast::<PyDict>()?;
                let event_type: String = dict
                    .get_item("event_type")?
                    .ok_or_else(|| ValidationError::new_err("missing 'event_type'"))?
                    .extract()?;
                let payload = py_to_value(
                    &dict
                        .get_item("payload")?
                        .ok_or_else(|| ValidationError::new_err("missing 'payload'"))?,
                )?;
                Ok(BatchEventEntry {
                    event_type,
                    payload,
                })
            })
            .collect::<PyResult<_>>()?;

        match self
            .inner
            .executor()
            .execute(Command::EventBatchAppend {
                branch: None,
                space: None,
                entries: batch,
            })
            .map_err(to_py_err)?
        {
            Output::BatchResults(results) => batch_results_to_py(py, results),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for EventBatchAppend",
            )),
        }
    }

    /// Batch set multiple state cells in a single transaction.
    ///
    /// Each entry should be a dict with 'cell' and 'value'.
    /// Returns a list of dicts with 'version' (on success) or 'error' (on failure).
    fn state_batch_set(
        &self,
        py: Python<'_>,
        entries: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let batch: Vec<BatchStateEntry> = entries
            .iter()
            .map(|item| {
                let dict = item.downcast::<PyDict>()?;
                let cell: String = dict
                    .get_item("cell")?
                    .ok_or_else(|| ValidationError::new_err("missing 'cell'"))?
                    .extract()?;
                let value = py_to_value(
                    &dict
                        .get_item("value")?
                        .ok_or_else(|| ValidationError::new_err("missing 'value'"))?,
                )?;
                Ok(BatchStateEntry { cell, value })
            })
            .collect::<PyResult<_>>()?;

        match self
            .inner
            .executor()
            .execute(Command::StateBatchSet {
                branch: None,
                space: None,
                entries: batch,
            })
            .map_err(to_py_err)?
        {
            Output::BatchResults(results) => batch_results_to_py(py, results),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for StateBatchSet",
            )),
        }
    }

    /// Batch set multiple JSON document paths in a single transaction.
    ///
    /// Each entry should be a dict with 'key', 'path', and 'value'.
    /// Returns a list of dicts with 'version' (on success) or 'error' (on failure).
    fn json_batch_set(
        &self,
        py: Python<'_>,
        entries: &Bound<'_, PyList>,
    ) -> PyResult<PyObject> {
        let batch: Vec<BatchJsonEntry> = entries
            .iter()
            .map(|item| {
                let dict = item.downcast::<PyDict>()?;
                let key: String = dict
                    .get_item("key")?
                    .ok_or_else(|| ValidationError::new_err("missing 'key'"))?
                    .extract()?;
                let path: String = dict
                    .get_item("path")?
                    .ok_or_else(|| ValidationError::new_err("missing 'path'"))?
                    .extract()?;
                let value = py_to_value(
                    &dict
                        .get_item("value")?
                        .ok_or_else(|| ValidationError::new_err("missing 'value'"))?,
                )?;
                Ok(BatchJsonEntry { key, path, value })
            })
            .collect::<PyResult<_>>()?;

        match self
            .inner
            .executor()
            .execute(Command::JsonBatchSet {
                branch: None,
                space: None,
                entries: batch,
            })
            .map_err(to_py_err)?
        {
            Output::BatchResults(results) => batch_results_to_py(py, results),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for JsonBatchSet",
            )),
        }
    }

    // =========================================================================
    // Branch Management
    // =========================================================================

    /// Get the current branch name.
    fn current_branch(&self) -> &str {
        self.inner.current_branch()
    }

    /// Switch to a different branch.
    fn set_branch(&mut self, branch: &str) -> PyResult<()> {
        self.inner.set_branch(branch).map_err(to_py_err)
    }

    /// Create a new empty branch.
    fn create_branch(&self, branch: &str) -> PyResult<()> {
        self.inner.create_branch(branch).map_err(to_py_err)
    }

    /// Fork the current branch to a new branch, copying all data.
    fn fork_branch(&self, py: Python<'_>, destination: &str) -> PyResult<PyObject> {
        let info = self.inner.fork_branch(destination).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("source", info.source)?;
        dict.set_item("destination", info.destination)?;
        dict.set_item("keys_copied", info.keys_copied)?;
        Ok(dict.unbind().into_any())
    }

    /// List all branches.
    fn list_branches(&self) -> PyResult<Vec<String>> {
        self.inner.list_branches().map_err(to_py_err)
    }

    /// Delete a branch.
    fn delete_branch(&self, branch: &str) -> PyResult<()> {
        self.inner.delete_branch(branch).map_err(to_py_err)
    }

    /// Check if a branch exists.
    fn branch_exists(&self, name: &str) -> PyResult<bool> {
        self.inner.branches().exists(name).map_err(to_py_err)
    }

    /// Get branch metadata with version info.
    ///
    /// Returns a dict with 'id', 'status', 'created_at', 'updated_at', 'version', 'timestamp',
    /// or None if the branch does not exist.
    fn branch_get(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match self.inner.branch_get(name).map_err(to_py_err)? {
            Some(info) => versioned_branch_info_to_py(py, info),
            None => Ok(py.None()),
        }
    }

    /// Compare two branches.
    fn diff_branches(
        &self,
        py: Python<'_>,
        branch_a: &str,
        branch_b: &str,
    ) -> PyResult<PyObject> {
        let diff = self
            .inner
            .diff_branches(branch_a, branch_b)
            .map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("branch_a", &diff.branch_a)?;
        dict.set_item("branch_b", &diff.branch_b)?;
        let summary = PyDict::new_bound(py);
        summary.set_item("total_added", diff.summary.total_added)?;
        summary.set_item("total_removed", diff.summary.total_removed)?;
        summary.set_item("total_modified", diff.summary.total_modified)?;
        dict.set_item("summary", summary)?;
        Ok(dict.unbind().into_any())
    }

    /// Merge a branch into the current branch.
    #[pyo3(signature = (source, strategy=None))]
    fn merge_branches(
        &self,
        py: Python<'_>,
        source: &str,
        strategy: Option<&str>,
    ) -> PyResult<PyObject> {
        let strat = match strategy.unwrap_or("last_writer_wins") {
            "last_writer_wins" => MergeStrategy::LastWriterWins,
            "strict" => MergeStrategy::Strict,
            _ => return Err(ValidationError::new_err("Invalid merge strategy")),
        };
        let target = self.inner.current_branch().to_string();
        let info = self.inner.merge_branches(source, &target, strat).map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("keys_applied", info.keys_applied)?;
        dict.set_item("spaces_merged", info.spaces_merged)?;
        let conflicts = PyList::empty_bound(py);
        for c in info.conflicts {
            let cdict = PyDict::new_bound(py);
            cdict.set_item("key", c.key)?;
            cdict.set_item("space", c.space)?;
            conflicts.append(cdict)?;
        }
        dict.set_item("conflicts", conflicts)?;
        Ok(dict.unbind().into_any())
    }

    // =========================================================================
    // Space Management
    // =========================================================================

    /// Get the current space name.
    fn current_space(&self) -> &str {
        self.inner.current_space()
    }

    /// Switch to a different space.
    fn set_space(&mut self, space: &str) -> PyResult<()> {
        self.inner.set_space(space).map_err(to_py_err)
    }

    /// List all spaces in the current branch.
    fn list_spaces(&self) -> PyResult<Vec<String>> {
        self.inner.list_spaces().map_err(to_py_err)
    }

    /// Delete a space and all its data.
    fn delete_space(&self, space: &str) -> PyResult<()> {
        self.inner.delete_space(space).map_err(to_py_err)
    }

    /// Force delete a space even if non-empty.
    fn delete_space_force(&self, space: &str) -> PyResult<()> {
        self.inner.delete_space_force(space).map_err(to_py_err)
    }

    // =========================================================================
    // Database Operations
    // =========================================================================

    /// Check database connectivity.
    fn ping(&self) -> PyResult<String> {
        self.inner.ping().map_err(to_py_err)
    }

    /// Get database info.
    fn info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let info = self.inner.info().map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("version", info.version)?;
        dict.set_item("uptime_secs", info.uptime_secs)?;
        dict.set_item("branch_count", info.branch_count)?;
        dict.set_item("total_keys", info.total_keys)?;
        Ok(dict.unbind().into_any())
    }

    /// Flush writes to disk.
    fn flush(&self) -> PyResult<()> {
        self.inner.flush().map_err(to_py_err)
    }

    /// Trigger compaction.
    fn compact(&self) -> PyResult<()> {
        self.inner.compact().map_err(to_py_err)
    }

    // =========================================================================
    // Bundle Operations
    // =========================================================================

    /// Export a branch to a bundle file.
    ///
    /// Returns a dict with 'branch_id', 'path', 'entry_count', 'bundle_size'.
    fn branch_export(&self, py: Python<'_>, branch: &str, path: &str) -> PyResult<PyObject> {
        let result = self.inner.branch_export(branch, path).map_err(to_py_err)?;
        branch_export_result_to_py(py, result)
    }

    /// Import a branch from a bundle file.
    ///
    /// Returns a dict with 'branch_id', 'transactions_applied', 'keys_written'.
    fn branch_import(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let result = self.inner.branch_import(path).map_err(to_py_err)?;
        branch_import_result_to_py(py, result)
    }

    /// Validate a bundle file without importing.
    ///
    /// Returns a dict with 'branch_id', 'format_version', 'entry_count', 'checksums_valid'.
    fn branch_validate_bundle(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let result = self
            .inner
            .branch_validate_bundle(path)
            .map_err(to_py_err)?;
        bundle_validate_result_to_py(py, result)
    }

    // =========================================================================
    // Transaction Operations (NEW)
    // =========================================================================

    /// Begin a new transaction.
    ///
    /// All subsequent data operations will be part of this transaction until
    /// commit() or rollback() is called.
    #[pyo3(signature = (read_only=None))]
    fn begin(&self, read_only: Option<bool>) -> PyResult<()> {
        let mut session_ref = self.session.lock()
            .map_err(|_| PyRuntimeError::new_err("session lock poisoned"))?;
        if session_ref.is_none() {
            *session_ref = Some(self.inner.session());
        }
        let session = session_ref.as_mut().unwrap();

        let cmd = Command::TxnBegin {
            branch: None,
            options: Some(::stratadb::TxnOptions {
                read_only: read_only.unwrap_or(false),
            }),
        };
        session.execute(cmd).map_err(to_py_err)?;
        Ok(())
    }

    /// Commit the current transaction.
    ///
    /// Returns the commit version number.
    fn commit(&self) -> PyResult<u64> {
        let mut session_ref = self.session.lock()
            .map_err(|_| PyRuntimeError::new_err("session lock poisoned"))?;
        let session = session_ref
            .as_mut()
            .ok_or_else(|| StateError::new_err("No transaction active"))?;

        match session.execute(Command::TxnCommit).map_err(to_py_err)? {
            Output::TxnCommitted { version } => Ok(version),
            _ => Err(PyRuntimeError::new_err("Unexpected output for TxnCommit")),
        }
    }

    /// Rollback the current transaction.
    fn rollback(&self) -> PyResult<()> {
        let mut session_ref = self.session.lock()
            .map_err(|_| PyRuntimeError::new_err("session lock poisoned"))?;
        let session = session_ref
            .as_mut()
            .ok_or_else(|| StateError::new_err("No transaction active"))?;

        session.execute(Command::TxnRollback).map_err(to_py_err)?;
        Ok(())
    }

    /// Get current transaction info.
    ///
    /// Returns a dict with 'id', 'status', 'started_at', or None if no transaction is active.
    fn txn_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut session_ref = self.session.lock()
            .map_err(|_| PyRuntimeError::new_err("session lock poisoned"))?;
        if session_ref.is_none() {
            return Ok(py.None());
        }
        let session = session_ref.as_mut().unwrap();

        match session.execute(Command::TxnInfo).map_err(to_py_err)? {
            Output::TxnInfo(Some(info)) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("id", info.id)?;
                dict.set_item("status", format!("{:?}", info.status).to_lowercase())?;
                dict.set_item("started_at", info.started_at)?;
                Ok(dict.unbind().into_any())
            }
            Output::TxnInfo(None) => Ok(py.None()),
            _ => Err(PyRuntimeError::new_err("Unexpected output for TxnInfo")),
        }
    }

    /// Check if a transaction is currently active.
    fn txn_is_active(&self) -> PyResult<bool> {
        let mut session_ref = self.session.lock()
            .map_err(|_| PyRuntimeError::new_err("session lock poisoned"))?;
        if session_ref.is_none() {
            return Ok(false);
        }
        let session = session_ref.as_mut().unwrap();

        match session.execute(Command::TxnIsActive).map_err(to_py_err)? {
            Output::Bool(active) => Ok(active),
            _ => Err(PyRuntimeError::new_err("Unexpected output for TxnIsActive")),
        }
    }

    // =========================================================================
    // Missing State Operations (NEW)
    // =========================================================================

    /// Delete a state cell.
    ///
    /// Returns True if the cell existed and was deleted, False otherwise.
    fn state_delete(&self, cell: &str) -> PyResult<bool> {
        match self
            .inner
            .executor()
            .execute(Command::StateDelete {
                branch: None,
                space: None,
                cell: cell.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Bool(deleted) => Ok(deleted),
            _ => Err(PyRuntimeError::new_err("Unexpected output for StateDelete")),
        }
    }

    /// List state cell names with optional prefix filter.
    ///
    /// Args:
    ///     prefix: Optional prefix to filter cell names.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (prefix=None, as_of=None))]
    fn state_list(&self, prefix: Option<&str>, as_of: Option<u64>) -> PyResult<Vec<String>> {
        match self
            .inner
            .executor()
            .execute(Command::StateList {
                branch: None,
                space: None,
                prefix: prefix.map(|s| s.to_string()),
                as_of,
            })
            .map_err(to_py_err)?
        {
            Output::Keys(keys) => Ok(keys),
            _ => Err(PyRuntimeError::new_err("Unexpected output for StateList")),
        }
    }

    // =========================================================================
    // Versioned Getters (NEW)
    // =========================================================================

    /// Get a value by key with version info.
    ///
    /// Returns a dict with 'value', 'version', 'timestamp', or None if key doesn't exist.
    fn kv_get_versioned(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.kv_getv(key).map_err(to_py_err)? {
            Some(versions) if !versions.is_empty() => {
                // Return the latest version
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap())?)
            }
            _ => Ok(py.None()),
        }
    }

    /// Get a state cell value with version info.
    ///
    /// Returns a dict with 'value', 'version', 'timestamp', or None if cell doesn't exist.
    fn state_get_versioned(&self, py: Python<'_>, cell: &str) -> PyResult<PyObject> {
        match self.inner.state_getv(cell).map_err(to_py_err)? {
            Some(versions) if !versions.is_empty() => {
                // Return the latest version
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap())?)
            }
            _ => Ok(py.None()),
        }
    }

    /// Get a JSON document value with version info.
    ///
    /// Returns a dict with 'value', 'version', 'timestamp', or None if key doesn't exist.
    fn json_get_versioned(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.json_getv(key).map_err(to_py_err)? {
            Some(versions) if !versions.is_empty() => {
                // Return the latest version
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap())?)
            }
            _ => Ok(py.None()),
        }
    }

    // =========================================================================
    // Pagination Improvements (NEW)
    // =========================================================================

    /// List keys with pagination support.
    ///
    /// Returns a dict with 'keys'.
    ///
    /// Note: Cursor-based pagination is not yet supported for KV lists.
    /// The underlying executor returns keys without a continuation cursor.
    /// Use `prefix` and `limit` to narrow results.
    ///
    /// Args:
    ///     prefix: Optional prefix to filter keys.
    ///     limit: Optional maximum number of keys to return.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (prefix=None, limit=None, as_of=None))]
    fn kv_list_paginated(
        &self,
        py: Python<'_>,
        prefix: Option<&str>,
        limit: Option<u64>,
        as_of: Option<u64>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::KvList {
                branch: None,
                space: None,
                prefix: prefix.map(|s| s.to_string()),
                cursor: None,
                limit,
                as_of,
            })
            .map_err(to_py_err)?
        {
            Output::Keys(keys) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("keys", keys)?;
                Ok(dict.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for KvList")),
        }
    }

    /// List events by type with pagination support.
    ///
    /// Returns a list of event dicts. Use `after` to paginate from a sequence number.
    ///
    /// Args:
    ///     event_type: The event type to filter by.
    ///     limit: Optional maximum number of events to return.
    ///     after: Optional sequence number to paginate from.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (event_type, limit=None, after=None, as_of=None))]
    fn event_list_paginated(
        &self,
        py: Python<'_>,
        event_type: &str,
        limit: Option<u64>,
        after: Option<u64>,
        as_of: Option<u64>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::EventGetByType {
                branch: None,
                space: None,
                event_type: event_type.to_string(),
                limit,
                after_sequence: after,
                as_of,
            })
            .map_err(to_py_err)?
        {
            Output::VersionedValues(events) => {
                let list = PyList::empty_bound(py);
                for vv in events {
                    list.append(versioned_to_py(py, vv)?)?;
                }
                Ok(list.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for EventGetByType",
            )),
        }
    }

    // =========================================================================
    // Enhanced Vector Search (NEW)
    // =========================================================================

    /// Search for similar vectors with optional filter and metric override.
    ///
    /// Args:
    ///     collection: The collection name.
    ///     query: Query vector (numpy array or list of floats).
    ///     k: Number of nearest neighbors to return.
    ///     metric: Optional distance metric override.
    ///     filter: Optional metadata filters.
    ///     as_of: Optional timestamp (microseconds since epoch) for time-travel reads.
    #[pyo3(signature = (collection, query, k, metric=None, filter=None, as_of=None))]
    fn vector_search_filtered(
        &self,
        py: Python<'_>,
        collection: &str,
        query: &Bound<'_, PyAny>,
        k: u64,
        metric: Option<&str>,
        filter: Option<&Bound<'_, PyList>>,
        as_of: Option<u64>,
    ) -> PyResult<PyObject> {
        let vec = extract_vector(query)?;

        let metric_enum = match metric {
            Some("cosine") => Some(DistanceMetric::Cosine),
            Some("euclidean") => Some(DistanceMetric::Euclidean),
            Some("dot_product") | Some("dotproduct") => Some(DistanceMetric::DotProduct),
            Some(m) => return Err(ValidationError::new_err(format!("Invalid metric: {}", m))),
            None => None,
        };

        let filter_vec = match filter {
            Some(list) => {
                let mut filters = Vec::new();
                for item in list.iter() {
                    let dict = item.downcast::<PyDict>()?;
                    let field: String = dict
                        .get_item("field")?
                        .ok_or_else(|| ValidationError::new_err("filter missing 'field'"))?
                        .extract()?;
                    let op_str: String = dict
                        .get_item("op")?
                        .ok_or_else(|| ValidationError::new_err("filter missing 'op'"))?
                        .extract()?;
                    let op = match op_str.as_str() {
                        "eq" => FilterOp::Eq,
                        "ne" => FilterOp::Ne,
                        "gt" => FilterOp::Gt,
                        "gte" => FilterOp::Gte,
                        "lt" => FilterOp::Lt,
                        "lte" => FilterOp::Lte,
                        "in" => FilterOp::In,
                        "contains" => FilterOp::Contains,
                        _ => {
                            return Err(ValidationError::new_err(format!(
                                "Invalid filter op: {}",
                                op_str
                            )))
                        }
                    };
                    let value_obj = dict
                        .get_item("value")?
                        .ok_or_else(|| ValidationError::new_err("filter missing 'value'"))?;
                    let value = py_to_value(&value_obj)?;
                    filters.push(MetadataFilter { field, op, value });
                }
                Some(filters)
            }
            None => None,
        };

        match self
            .inner
            .executor()
            .execute(Command::VectorSearch {
                branch: None,
                space: None,
                collection: collection.to_string(),
                query: vec,
                k,
                filter: filter_vec,
                metric: metric_enum,
                as_of,
            })
            .map_err(to_py_err)?
        {
            Output::VectorMatches(matches) => {
                let list = PyList::empty_bound(py);
                for m in matches {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("key", m.key)?;
                    dict.set_item("score", m.score)?;
                    if let Some(meta) = m.metadata {
                        dict.set_item("metadata", value_to_py(py, meta)?)?;
                    }
                    list.append(dict)?;
                }
                Ok(list.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for VectorSearch",
            )),
        }
    }

    // =========================================================================
    // Space Operations (NEW)
    // =========================================================================

    /// Create a new space explicitly.
    ///
    /// Spaces are auto-created on first write, but this allows pre-creation.
    fn space_create(&self, space: &str) -> PyResult<()> {
        match self
            .inner
            .executor()
            .execute(Command::SpaceCreate {
                branch: None,
                space: space.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Unit => Ok(()),
            _ => Err(PyRuntimeError::new_err("Unexpected output for SpaceCreate")),
        }
    }

    /// Check if a space exists in the current branch.
    fn space_exists(&self, space: &str) -> PyResult<bool> {
        match self
            .inner
            .executor()
            .execute(Command::SpaceExists {
                branch: None,
                space: space.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Bool(exists) => Ok(exists),
            _ => Err(PyRuntimeError::new_err("Unexpected output for SpaceExists")),
        }
    }

    // =========================================================================
    // Model Configuration
    // =========================================================================

    /// Get the current database configuration.
    ///
    /// Returns a dict with 'durability', 'auto_embed', 'embed_model', and optional 'model'.
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let cfg = self.inner.config().map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("durability", &cfg.durability)?;
        dict.set_item("auto_embed", cfg.auto_embed)?;
        dict.set_item("embed_model", &cfg.embed_model)?;
        if let Some(ref model) = cfg.model {
            let model_dict = PyDict::new_bound(py);
            model_dict.set_item("endpoint", &model.endpoint)?;
            model_dict.set_item("model", &model.model)?;
            match &model.api_key {
                Some(key) => model_dict.set_item("api_key", key)?,
                None => model_dict.set_item("api_key", py.None())?,
            };
            model_dict.set_item("timeout_ms", model.timeout_ms)?;
            dict.set_item("model", model_dict)?;
        } else {
            dict.set_item("model", py.None())?;
        }
        Ok(dict.unbind().into_any())
    }

    /// Get a snapshot of the embedding pipeline status.
    ///
    /// Returns a dict with:
    ///   - auto_embed (bool): whether auto-embedding is enabled
    ///   - pending (int): items waiting in the buffer
    ///   - total_queued (int): cumulative items pushed
    ///   - total_embedded (int): cumulative items successfully embedded
    ///   - total_failed (int): cumulative items that failed
    ///   - is_idle (bool): True when no work is pending or in-flight
    fn embed_status(&self, py: Python<'_>) -> PyResult<PyObject> {
        let info = self.inner.embed_status().map_err(to_py_err)?;
        let dict = PyDict::new_bound(py);
        dict.set_item("auto_embed", info.auto_embed)?;
        dict.set_item("batch_size", info.batch_size)?;
        dict.set_item("pending", info.pending)?;
        dict.set_item("total_queued", info.total_queued)?;
        dict.set_item("total_embedded", info.total_embedded)?;
        dict.set_item("total_failed", info.total_failed)?;
        dict.set_item("scheduler_queue_depth", info.scheduler_queue_depth)?;
        dict.set_item("scheduler_active_tasks", info.scheduler_active_tasks)?;
        dict.set_item("is_idle", info.pending == 0
            && info.scheduler_active_tasks == 0
            && info.scheduler_queue_depth == 0)?;
        Ok(dict.unbind().into_any())
    }

    /// Check whether auto-embedding is enabled.
    fn auto_embed_enabled(&self) -> PyResult<bool> {
        self.inner.auto_embed_enabled().map_err(to_py_err)
    }

    /// Enable or disable auto-embedding of text values.
    ///
    /// Persisted to strata.toml for disk-backed databases.
    fn set_auto_embed(&self, enabled: bool) -> PyResult<()> {
        self.inner.set_auto_embed(enabled).map_err(to_py_err)
    }

    /// Configure an inference model endpoint for intelligent search.
    ///
    /// When a model is configured, `search()` transparently expands queries
    /// using the model for better recall. Search works identically without a model;
    /// the model is an invisible accelerator.
    ///
    /// Args:
    ///     endpoint: OpenAI-compatible API endpoint URL (e.g. "http://localhost:11434/v1").
    ///     model: Model name (e.g. "qwen3:1.7b").
    ///     api_key: Optional bearer token for the endpoint.
    ///     timeout_ms: Request timeout in milliseconds (default: 5000).
    #[pyo3(signature = (endpoint, model, api_key=None, timeout_ms=None))]
    fn configure_model(
        &self,
        endpoint: &str,
        model: &str,
        api_key: Option<&str>,
        timeout_ms: Option<u64>,
    ) -> PyResult<()> {
        self.inner
            .executor()
            .execute(Command::ConfigureModel {
                endpoint: endpoint.to_string(),
                model: model.to_string(),
                api_key: api_key.map(|s| s.to_string()),
                timeout_ms,
            })
            .map_err(to_py_err)?;
        Ok(())
    }

    // =========================================================================
    // Search (NEW)
    // =========================================================================

    /// Search across multiple primitives for matching content.
    ///
    /// Returns a list of dicts with 'entity', 'primitive', 'score', 'rank', 'snippet'.
    #[pyo3(signature = (query, k=None, primitives=None, time_range=None, mode=None, expand=None, rerank=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        k: Option<u64>,
        primitives: Option<Vec<String>>,
        time_range: Option<&Bound<'_, PyDict>>,
        mode: Option<&str>,
        expand: Option<bool>,
        rerank: Option<bool>,
    ) -> PyResult<PyObject> {
        let tr = if let Some(dict) = time_range {
            let start: String = dict
                .get_item("start")?
                .ok_or_else(|| PyRuntimeError::new_err("time_range missing 'start' key"))?
                .extract()?;
            let end: String = dict
                .get_item("end")?
                .ok_or_else(|| PyRuntimeError::new_err("time_range missing 'end' key"))?
                .extract()?;
            Some(TimeRangeInput { start, end })
        } else {
            None
        };

        let sq = SearchQuery {
            query: query.to_string(),
            k,
            primitives,
            time_range: tr,
            mode: mode.map(|s| s.to_string()),
            expand,
            rerank,
            precomputed_embedding: None,
        };

        let cmd = Command::Search {
            branch: None,
            space: None,
            search: sq,
        };

        // Clone executor ref before releasing GIL
        let executor = self.inner.executor();

        // Release GIL during Rust execution to allow parallel Python threads
        let result = py.allow_threads(|| executor.execute(cmd)).map_err(to_py_err)?;

        match result {
            Output::SearchResults(results) => {
                let list = PyList::empty_bound(py);
                for hit in results {
                    let dict = PyDict::new_bound(py);
                    dict.set_item("entity", hit.entity)?;
                    dict.set_item("primitive", hit.primitive)?;
                    dict.set_item("score", hit.score)?;
                    dict.set_item("rank", hit.rank)?;
                    if let Some(snippet) = hit.snippet {
                        dict.set_item("snippet", snippet)?;
                    }
                    list.append(dict)?;
                }
                Ok(list.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for Search")),
        }
    }

    // =========================================================================
    // Time-Travel
    // =========================================================================

    /// Get the time range of data in the current branch.
    ///
    /// Returns a dict with 'oldest_ts' and 'latest_ts' (microseconds since epoch),
    /// or None values if the branch has no data.
    fn time_range(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self.inner.executor().execute(Command::TimeRange {
            branch: None,
        }).map_err(to_py_err)? {
            Output::TimeRange { oldest_ts, latest_ts } => {
                let dict = PyDict::new_bound(py);
                match oldest_ts {
                    Some(ts) => dict.set_item("oldest_ts", ts)?,
                    None => dict.set_item("oldest_ts", py.None())?,
                };
                match latest_ts {
                    Some(ts) => dict.set_item("latest_ts", ts)?,
                    None => dict.set_item("latest_ts", py.None())?,
                };
                Ok(dict.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for TimeRange")),
        }
    }

    // =========================================================================
    // Embedding
    // =========================================================================

    /// Embed a single text string into a vector.
    ///
    /// Returns a numpy array of float32 values.
    fn embed(&self, py: Python<'_>, text: &str) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::Embed {
                text: text.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Embedding(vec) => {
                let arr = PyArray1::from_vec_bound(py, vec);
                Ok(arr.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for Embed")),
        }
    }

    /// Embed multiple texts into vectors.
    ///
    /// Returns a list of numpy arrays.
    fn embed_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::EmbedBatch { texts })
            .map_err(to_py_err)?
        {
            Output::Embeddings(vecs) => {
                let list = PyList::empty_bound(py);
                for vec in vecs {
                    let arr = PyArray1::from_vec_bound(py, vec);
                    list.append(arr)?;
                }
                Ok(list.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for EmbedBatch")),
        }
    }

    // =========================================================================
    // Text Generation
    // =========================================================================

    /// Generate text using a loaded model.
    ///
    /// Args:
    ///     model: Model name (e.g. "qwen3:1.7b").
    ///     prompt: The prompt to generate from.
    ///     max_tokens: Maximum number of tokens to generate.
    ///     temperature: Sampling temperature (0.0 = greedy).
    ///     top_k: Top-k sampling parameter.
    ///     top_p: Top-p (nucleus) sampling parameter.
    ///     seed: Random seed for reproducibility.
    ///     stop_tokens: Token IDs that stop generation.
    ///
    /// Returns a dict with 'text', 'stop_reason', 'prompt_tokens',
    /// 'completion_tokens', 'model'.
    #[pyo3(signature = (model, prompt, max_tokens=None, temperature=None, top_k=None, top_p=None, seed=None, stop_tokens=None))]
    fn generate(
        &self,
        py: Python<'_>,
        model: &str,
        prompt: &str,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_k: Option<usize>,
        top_p: Option<f32>,
        seed: Option<u64>,
        stop_tokens: Option<Vec<u32>>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::Generate {
                model: model.to_string(),
                prompt: prompt.to_string(),
                max_tokens,
                temperature,
                top_k,
                top_p,
                seed,
                stop_tokens,
            })
            .map_err(to_py_err)?
        {
            Output::Generated(result) => generation_result_to_py(py, result),
            _ => Err(PyRuntimeError::new_err("Unexpected output for Generate")),
        }
    }

    /// Tokenize text into token IDs.
    ///
    /// Args:
    ///     model: Model name whose tokenizer to use.
    ///     text: Text to tokenize.
    ///     add_special_tokens: Whether to add special tokens (default: true).
    ///
    /// Returns a dict with 'ids' (list of ints), 'count' (int), 'model' (str).
    #[pyo3(signature = (model, text, add_special_tokens=None))]
    fn tokenize(
        &self,
        py: Python<'_>,
        model: &str,
        text: &str,
        add_special_tokens: Option<bool>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::Tokenize {
                model: model.to_string(),
                text: text.to_string(),
                add_special_tokens,
            })
            .map_err(to_py_err)?
        {
            Output::TokenIds(result) => tokenize_result_to_py(py, result),
            _ => Err(PyRuntimeError::new_err("Unexpected output for Tokenize")),
        }
    }

    /// Convert token IDs back to text.
    ///
    /// Args:
    ///     model: Model name whose tokenizer to use.
    ///     ids: List of token IDs.
    ///
    /// Returns the decoded text string.
    fn detokenize(&self, model: &str, ids: Vec<u32>) -> PyResult<String> {
        match self
            .inner
            .executor()
            .execute(Command::Detokenize {
                model: model.to_string(),
                ids,
            })
            .map_err(to_py_err)?
        {
            Output::Text(text) => Ok(text),
            _ => Err(PyRuntimeError::new_err("Unexpected output for Detokenize")),
        }
    }

    /// Unload a model from memory.
    ///
    /// Returns True if the model was loaded and is now unloaded.
    fn generate_unload(&self, model: &str) -> PyResult<bool> {
        match self
            .inner
            .executor()
            .execute(Command::GenerateUnload {
                model: model.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Bool(unloaded) => Ok(unloaded),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for GenerateUnload",
            )),
        }
    }

    // =========================================================================
    // Model Management
    // =========================================================================

    /// List available models from the registry.
    ///
    /// Returns a list of dicts with 'name', 'task', 'architecture',
    /// 'default_quant', 'embedding_dim', 'is_local', 'size_bytes'.
    fn models_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::ModelsList)
            .map_err(to_py_err)?
        {
            Output::ModelsList(models) => models_list_to_py(py, models),
            _ => Err(PyRuntimeError::new_err("Unexpected output for ModelsList")),
        }
    }

    /// Download a model by name.
    ///
    /// Returns a dict with 'name' and 'path'.
    fn models_pull(&self, py: Python<'_>, name: &str) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::ModelsPull {
                name: name.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::ModelsPulled { name, path } => {
                let dict = PyDict::new_bound(py);
                dict.set_item("name", name)?;
                dict.set_item("path", path)?;
                Ok(dict.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for ModelsPull")),
        }
    }

    /// List locally downloaded models.
    ///
    /// Returns a list of dicts with 'name', 'task', 'architecture',
    /// 'default_quant', 'embedding_dim', 'is_local', 'size_bytes'.
    fn models_local(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::ModelsLocal)
            .map_err(to_py_err)?
        {
            Output::ModelsList(models) => models_list_to_py(py, models),
            _ => Err(PyRuntimeError::new_err("Unexpected output for ModelsLocal")),
        }
    }

    // =========================================================================
    // Durability
    // =========================================================================

    /// Get WAL durability counters.
    ///
    /// Returns a dict with 'wal_appends', 'sync_calls', 'bytes_written', 'sync_nanos'.
    fn durability_counters(&self, py: Python<'_>) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::DurabilityCounters)
            .map_err(to_py_err)?
        {
            Output::DurabilityCounters(counters) => wal_counters_to_py(py, counters),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for DurabilityCounters",
            )),
        }
    }

    // =========================================================================
    // Configuration Key Management
    // =========================================================================

    /// Set a database configuration key.
    ///
    /// Supported keys: `provider`, `default_model`, `anthropic_api_key`,
    /// `openai_api_key`, `google_api_key`, `embed_model`.
    fn configure_set(&self, key: &str, value: &str) -> PyResult<()> {
        match self
            .inner
            .executor()
            .execute(Command::ConfigureSet {
                key: key.to_string(),
                value: value.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::Unit => Ok(()),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for ConfigureSet",
            )),
        }
    }

    /// Get a database configuration value by key.
    ///
    /// Returns the value as a string, or None if not set.
    fn configure_get(&self, key: &str) -> PyResult<Option<String>> {
        match self
            .inner
            .executor()
            .execute(Command::ConfigureGetKey {
                key: key.to_string(),
            })
            .map_err(to_py_err)?
        {
            Output::ConfigValue(v) => Ok(v),
            _ => Err(PyRuntimeError::new_err(
                "Unexpected output for ConfigureGetKey",
            )),
        }
    }

    // =========================================================================
    // Retention (#8)
    // =========================================================================

    /// Apply retention policy (trigger garbage collection).
    fn retention_apply(&self) -> PyResult<()> {
        self.inner
            .executor()
            .execute(Command::RetentionApply { branch: None })
            .map_err(to_py_err)?;
        Ok(())
    }
}

/// Extract a vector from either a numpy array or a Python list.
fn extract_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    // Try numpy array first
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        return Ok(arr.to_vec()?);
    }
    // Try list of floats
    if let Ok(list) = obj.downcast::<PyList>() {
        let vec: PyResult<Vec<f32>> = list.iter().map(|item| item.extract::<f32>()).collect();
        return vec;
    }
    Err(ValidationError::new_err(
        "Expected numpy array or list of floats",
    ))
}

/// Convert batch results to a Python list of dicts.
fn batch_results_to_py(py: Python<'_>, results: Vec<BatchItemResult>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for r in results {
        let dict = PyDict::new_bound(py);
        match r.version {
            Some(v) => dict.set_item("version", v)?,
            None => dict.set_item("version", py.None())?,
        };
        match r.error {
            Some(e) => dict.set_item("error", e)?,
            None => dict.set_item("error", py.None())?,
        };
        list.append(dict)?;
    }
    Ok(list.unbind().into_any())
}

/// Convert CollectionInfo to Python dict.
fn collection_info_to_py(py: Python<'_>, c: CollectionInfo) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", c.name)?;
    dict.set_item("dimension", c.dimension)?;
    dict.set_item("metric", format!("{:?}", c.metric).to_lowercase())?;
    dict.set_item("count", c.count)?;
    dict.set_item("index_type", c.index_type)?;
    dict.set_item("memory_bytes", c.memory_bytes)?;
    Ok(dict.unbind().into_any())
}

/// Convert VersionedBranchInfo to Python dict.
fn versioned_branch_info_to_py(py: Python<'_>, info: VersionedBranchInfo) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.info.id.as_str())?;
    dict.set_item("status", format!("{:?}", info.info.status).to_lowercase())?;
    dict.set_item("created_at", info.info.created_at)?;
    dict.set_item("updated_at", info.info.updated_at)?;
    if let Some(parent) = info.info.parent_id {
        dict.set_item("parent_id", parent.as_str())?;
    }
    dict.set_item("version", info.version)?;
    dict.set_item("timestamp", info.timestamp)?;
    Ok(dict.unbind().into_any())
}

/// Convert BranchExportResult to Python dict.
fn branch_export_result_to_py(py: Python<'_>, r: BranchExportResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id)?;
    dict.set_item("path", r.path)?;
    dict.set_item("entry_count", r.entry_count)?;
    dict.set_item("bundle_size", r.bundle_size)?;
    Ok(dict.unbind().into_any())
}

/// Convert BranchImportResult to Python dict.
fn branch_import_result_to_py(py: Python<'_>, r: BranchImportResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id)?;
    dict.set_item("transactions_applied", r.transactions_applied)?;
    dict.set_item("keys_written", r.keys_written)?;
    Ok(dict.unbind().into_any())
}

/// Convert BundleValidateResult to Python dict.
fn bundle_validate_result_to_py(py: Python<'_>, r: BundleValidateResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id)?;
    dict.set_item("format_version", r.format_version)?;
    dict.set_item("entry_count", r.entry_count)?;
    dict.set_item("checksums_valid", r.checksums_valid)?;
    Ok(dict.unbind().into_any())
}

/// Convert GenerationResult to Python dict.
fn generation_result_to_py(py: Python<'_>, r: GenerationResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("text", r.text)?;
    dict.set_item("stop_reason", r.stop_reason)?;
    dict.set_item("prompt_tokens", r.prompt_tokens)?;
    dict.set_item("completion_tokens", r.completion_tokens)?;
    dict.set_item("model", r.model)?;
    Ok(dict.unbind().into_any())
}

/// Convert TokenizeResult to Python dict.
fn tokenize_result_to_py(py: Python<'_>, r: TokenizeResult) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("ids", r.ids)?;
    dict.set_item("count", r.count)?;
    dict.set_item("model", r.model)?;
    Ok(dict.unbind().into_any())
}

/// Convert a list of ModelInfoOutput to Python list of dicts.
fn models_list_to_py(py: Python<'_>, models: Vec<ModelInfoOutput>) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for m in models {
        let dict = PyDict::new_bound(py);
        dict.set_item("name", m.name)?;
        dict.set_item("task", m.task)?;
        dict.set_item("architecture", m.architecture)?;
        dict.set_item("default_quant", m.default_quant)?;
        dict.set_item("embedding_dim", m.embedding_dim)?;
        dict.set_item("is_local", m.is_local)?;
        dict.set_item("size_bytes", m.size_bytes)?;
        list.append(dict)?;
    }
    Ok(list.unbind().into_any())
}

/// Convert WalCounters to Python dict.
fn wal_counters_to_py(py: Python<'_>, c: WalCounters) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("wal_appends", c.wal_appends)?;
    dict.set_item("sync_calls", c.sync_calls)?;
    dict.set_item("bytes_written", c.bytes_written)?;
    dict.set_item("sync_nanos", c.sync_nanos)?;
    Ok(dict.unbind().into_any())
}

/// Python module initialization.
#[pymodule]
fn _stratadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStrata>()?;

    // Register exception hierarchy
    m.add("StrataError", m.py().get_type_bound::<StrataError_>())?;
    m.add("NotFoundError", m.py().get_type_bound::<NotFoundError>())?;
    m.add("ValidationError", m.py().get_type_bound::<ValidationError>())?;
    m.add("ConflictError", m.py().get_type_bound::<ConflictError>())?;
    m.add("StateError", m.py().get_type_bound::<StateError>())?;
    m.add("ConstraintError", m.py().get_type_bound::<ConstraintError>())?;
    m.add("AccessDeniedError", m.py().get_type_bound::<AccessDeniedError>())?;
    m.add("IoError", m.py().get_type_bound::<IoError>())?;

    Ok(())
}
