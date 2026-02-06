//! Python bindings for StrataDB.
//!
//! This module exposes the StrataDB API to Python via PyO3.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::cell::RefCell;
use std::collections::HashMap;

use ::stratadb::{
    BatchVectorEntry, BranchExportResult, BranchImportResult, BundleValidateResult,
    CollectionInfo, Command, DistanceMetric, Error as StrataError, FilterOp, MergeStrategy,
    MetadataFilter, Output, Session, Strata as RustStrata, Value, VersionedBranchInfo,
    VersionedValue,
};

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
        Err(PyValueError::new_err("Unsupported value type"))
    }
}

/// Convert a stratadb Value to a Python object.
fn value_to_py(py: Python<'_>, value: Value) -> PyObject {
    use pyo3::conversion::ToPyObject;
    match value {
        Value::Null => py.None(),
        Value::Bool(b) => b.to_object(py),
        Value::Int(i) => i.to_object(py),
        Value::Float(f) => f.to_object(py),
        Value::String(s) => s.to_object(py),
        Value::Bytes(b) => b.to_object(py),
        Value::Array(arr) => {
            let list = PyList::empty_bound(py);
            for item in arr {
                list.append(value_to_py(py, item)).unwrap();
            }
            list.unbind().into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new_bound(py);
            for (k, v) in map {
                dict.set_item(k, value_to_py(py, v)).unwrap();
            }
            dict.unbind().into_any()
        }
    }
}

/// Convert a VersionedValue to a Python dict.
fn versioned_to_py(py: Python<'_>, vv: VersionedValue) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("value", value_to_py(py, vv.value)).unwrap();
    dict.set_item("version", vv.version).unwrap();
    dict.set_item("timestamp", vv.timestamp).unwrap();
    dict.unbind().into_any()
}

/// Convert stratadb error to PyErr.
fn to_py_err(e: StrataError) -> PyErr {
    PyRuntimeError::new_err(format!("{}", e))
}

/// StrataDB database handle.
///
/// This is the main entry point for interacting with StrataDB from Python.
/// All operations go through this class.
#[pyclass(name = "Strata")]
pub struct PyStrata {
    inner: RustStrata,
    /// Session for transaction support. Lazily initialized.
    session: RefCell<Option<Session>>,
}

#[pymethods]
impl PyStrata {
    /// Open a database at the given path.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = RustStrata::open(path).map_err(to_py_err)?;
        Ok(Self {
            inner,
            session: RefCell::new(None),
        })
    }

    /// Create an in-memory database (no persistence).
    #[staticmethod]
    fn cache() -> PyResult<Self> {
        let inner = RustStrata::cache().map_err(to_py_err)?;
        Ok(Self {
            inner,
            session: RefCell::new(None),
        })
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
    fn kv_get(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.kv_get(key).map_err(to_py_err)? {
            Some(v) => Ok(value_to_py(py, v)),
            None => Ok(py.None()),
        }
    }

    /// Delete a key.
    fn kv_delete(&self, key: &str) -> PyResult<bool> {
        self.inner.kv_delete(key).map_err(to_py_err)
    }

    /// List keys with optional prefix filter.
    #[pyo3(signature = (prefix=None))]
    fn kv_list(&self, prefix: Option<&str>) -> PyResult<Vec<String>> {
        self.inner.kv_list(prefix).map_err(to_py_err)
    }

    /// Get version history for a key.
    fn kv_history(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        match self.inner.kv_getv(key).map_err(to_py_err)? {
            Some(versions) => {
                let list = PyList::empty_bound(py);
                for vv in versions {
                    list.append(versioned_to_py(py, vv))?;
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
    fn state_get(&self, py: Python<'_>, cell: &str) -> PyResult<PyObject> {
        match self.inner.state_get(cell).map_err(to_py_err)? {
            Some(v) => Ok(value_to_py(py, v)),
            None => Ok(py.None()),
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
                    list.append(versioned_to_py(py, vv))?;
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
    fn event_get(&self, py: Python<'_>, sequence: u64) -> PyResult<PyObject> {
        match self.inner.event_get(sequence).map_err(to_py_err)? {
            Some(vv) => Ok(versioned_to_py(py, vv)),
            None => Ok(py.None()),
        }
    }

    /// List events by type.
    fn event_list(&self, py: Python<'_>, event_type: &str) -> PyResult<PyObject> {
        let events = self.inner.event_get_by_type(event_type).map_err(to_py_err)?;
        let list = PyList::empty_bound(py);
        for vv in events {
            list.append(versioned_to_py(py, vv))?;
        }
        Ok(list.unbind().into_any())
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
    fn json_get(&self, py: Python<'_>, key: &str, path: &str) -> PyResult<PyObject> {
        match self.inner.json_get(key, path).map_err(to_py_err)? {
            Some(v) => Ok(value_to_py(py, v)),
            None => Ok(py.None()),
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
                    list.append(versioned_to_py(py, vv))?;
                }
                Ok(list.unbind().into_any())
            }
            None => Ok(py.None()),
        }
    }

    /// List JSON document keys.
    #[pyo3(signature = (limit, prefix=None, cursor=None))]
    fn json_list(
        &self,
        py: Python<'_>,
        limit: u64,
        prefix: Option<&str>,
        cursor: Option<&str>,
    ) -> PyResult<PyObject> {
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
            _ => return Err(PyValueError::new_err("Invalid metric")),
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
            let dict = collection_info_to_py(py, c);
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
    fn vector_get(&self, py: Python<'_>, collection: &str, key: &str) -> PyResult<PyObject> {
        match self.inner.vector_get(collection, key).map_err(to_py_err)? {
            Some(vd) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("key", &vd.key)?;
                let arr = PyArray1::from_slice_bound(py, &vd.data.embedding);
                dict.set_item("embedding", arr)?;
                if let Some(meta) = vd.data.metadata {
                    dict.set_item("metadata", value_to_py(py, meta))?;
                }
                dict.set_item("version", vd.version)?;
                dict.set_item("timestamp", vd.timestamp)?;
                Ok(dict.unbind().into_any())
            }
            None => Ok(py.None()),
        }
    }

    /// Delete a vector.
    fn vector_delete(&self, collection: &str, key: &str) -> PyResult<bool> {
        self.inner.vector_delete(collection, key).map_err(to_py_err)
    }

    /// Search for similar vectors.
    fn vector_search(
        &self,
        py: Python<'_>,
        collection: &str,
        query: &Bound<'_, PyAny>,
        k: u64,
    ) -> PyResult<PyObject> {
        let vec = extract_vector(query)?;
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
                dict.set_item("metadata", value_to_py(py, meta))?;
            }
            list.append(dict)?;
        }
        Ok(list.unbind().into_any())
    }

    /// Get statistics for a single collection.
    fn vector_collection_stats(&self, py: Python<'_>, collection: &str) -> PyResult<PyObject> {
        let info = self
            .inner
            .vector_collection_stats(collection)
            .map_err(to_py_err)?;
        Ok(collection_info_to_py(py, info))
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
                    .ok_or_else(|| PyValueError::new_err("missing 'key'"))?
                    .extract()?;
                let vec = extract_vector(
                    &dict
                        .get_item("vector")?
                        .ok_or_else(|| PyValueError::new_err("missing 'vector'"))?,
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
            Some(info) => Ok(versioned_branch_info_to_py(py, info)),
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
            _ => return Err(PyValueError::new_err("Invalid merge strategy")),
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
        Ok(branch_export_result_to_py(py, result))
    }

    /// Import a branch from a bundle file.
    ///
    /// Returns a dict with 'branch_id', 'transactions_applied', 'keys_written'.
    fn branch_import(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let result = self.inner.branch_import(path).map_err(to_py_err)?;
        Ok(branch_import_result_to_py(py, result))
    }

    /// Validate a bundle file without importing.
    ///
    /// Returns a dict with 'branch_id', 'format_version', 'entry_count', 'checksums_valid'.
    fn branch_validate_bundle(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        let result = self
            .inner
            .branch_validate_bundle(path)
            .map_err(to_py_err)?;
        Ok(bundle_validate_result_to_py(py, result))
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
        let mut session_ref = self.session.borrow_mut();
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
        let mut session_ref = self.session.borrow_mut();
        let session = session_ref
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transaction active"))?;

        match session.execute(Command::TxnCommit).map_err(to_py_err)? {
            Output::TxnCommitted { version } => Ok(version),
            _ => Err(PyRuntimeError::new_err("Unexpected output for TxnCommit")),
        }
    }

    /// Rollback the current transaction.
    fn rollback(&self) -> PyResult<()> {
        let mut session_ref = self.session.borrow_mut();
        let session = session_ref
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err("No transaction active"))?;

        session.execute(Command::TxnRollback).map_err(to_py_err)?;
        Ok(())
    }

    /// Get current transaction info.
    ///
    /// Returns a dict with 'id', 'status', 'started_at', or None if no transaction is active.
    fn txn_info(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut session_ref = self.session.borrow_mut();
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
        let mut session_ref = self.session.borrow_mut();
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
    #[pyo3(signature = (prefix=None))]
    fn state_list(&self, prefix: Option<&str>) -> PyResult<Vec<String>> {
        match self
            .inner
            .executor()
            .execute(Command::StateList {
                branch: None,
                space: None,
                prefix: prefix.map(|s| s.to_string()),
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
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap()))
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
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap()))
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
                Ok(versioned_to_py(py, versions.into_iter().next().unwrap()))
            }
            _ => Ok(py.None()),
        }
    }

    // =========================================================================
    // Pagination Improvements (NEW)
    // =========================================================================

    /// List keys with pagination support.
    ///
    /// Returns a dict with 'keys' and optionally 'cursor' for the next page.
    #[pyo3(signature = (prefix=None, limit=None, cursor=None))]
    fn kv_list_paginated(
        &self,
        py: Python<'_>,
        prefix: Option<&str>,
        limit: Option<u64>,
        cursor: Option<&str>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::KvList {
                branch: None,
                space: None,
                prefix: prefix.map(|s| s.to_string()),
                cursor: cursor.map(|s| s.to_string()),
                limit,
            })
            .map_err(to_py_err)?
        {
            Output::Keys(keys) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("keys", keys)?;
                // KvList returns Output::Keys which doesn't include cursor
                // Cursor-based pagination for KV is not yet fully implemented in core
                Ok(dict.unbind().into_any())
            }
            _ => Err(PyRuntimeError::new_err("Unexpected output for KvList")),
        }
    }

    /// List events by type with pagination support.
    ///
    /// Returns a list of event dicts. Use `after` to paginate from a sequence number.
    #[pyo3(signature = (event_type, limit=None, after=None))]
    fn event_list_paginated(
        &self,
        py: Python<'_>,
        event_type: &str,
        limit: Option<u64>,
        after: Option<u64>,
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
            })
            .map_err(to_py_err)?
        {
            Output::VersionedValues(events) => {
                let list = PyList::empty_bound(py);
                for vv in events {
                    list.append(versioned_to_py(py, vv))?;
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
    #[pyo3(signature = (collection, query, k, metric=None, filter=None))]
    fn vector_search_filtered(
        &self,
        py: Python<'_>,
        collection: &str,
        query: &Bound<'_, PyAny>,
        k: u64,
        metric: Option<&str>,
        filter: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyObject> {
        let vec = extract_vector(query)?;

        let metric_enum = match metric {
            Some("cosine") => Some(DistanceMetric::Cosine),
            Some("euclidean") => Some(DistanceMetric::Euclidean),
            Some("dot_product") | Some("dotproduct") => Some(DistanceMetric::DotProduct),
            Some(m) => return Err(PyValueError::new_err(format!("Invalid metric: {}", m))),
            None => None,
        };

        let filter_vec = match filter {
            Some(list) => {
                let mut filters = Vec::new();
                for item in list.iter() {
                    let dict = item.downcast::<PyDict>()?;
                    let field: String = dict
                        .get_item("field")?
                        .ok_or_else(|| PyValueError::new_err("filter missing 'field'"))?
                        .extract()?;
                    let op_str: String = dict
                        .get_item("op")?
                        .ok_or_else(|| PyValueError::new_err("filter missing 'op'"))?
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
                            return Err(PyValueError::new_err(format!(
                                "Invalid filter op: {}",
                                op_str
                            )))
                        }
                    };
                    let value_obj = dict
                        .get_item("value")?
                        .ok_or_else(|| PyValueError::new_err("filter missing 'value'"))?;
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
                        dict.set_item("metadata", value_to_py(py, meta))?;
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
    // Search (NEW)
    // =========================================================================

    /// Search across multiple primitives for matching content.
    ///
    /// Returns a list of dicts with 'entity', 'primitive', 'score', 'rank', 'snippet'.
    #[pyo3(signature = (query, k=None, primitives=None))]
    fn search(
        &self,
        py: Python<'_>,
        query: &str,
        k: Option<u64>,
        primitives: Option<Vec<String>>,
    ) -> PyResult<PyObject> {
        match self
            .inner
            .executor()
            .execute(Command::Search {
                branch: None,
                space: None,
                query: query.to_string(),
                k,
                primitives,
            })
            .map_err(to_py_err)?
        {
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
    Err(PyValueError::new_err(
        "Expected numpy array or list of floats",
    ))
}

/// Convert CollectionInfo to Python dict.
fn collection_info_to_py(py: Python<'_>, c: CollectionInfo) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("name", c.name).unwrap();
    dict.set_item("dimension", c.dimension).unwrap();
    dict.set_item("metric", format!("{:?}", c.metric).to_lowercase())
        .unwrap();
    dict.set_item("count", c.count).unwrap();
    dict.set_item("index_type", c.index_type).unwrap();
    dict.set_item("memory_bytes", c.memory_bytes).unwrap();
    dict.unbind().into_any()
}

/// Convert VersionedBranchInfo to Python dict.
fn versioned_branch_info_to_py(py: Python<'_>, info: VersionedBranchInfo) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("id", info.info.id.as_str()).unwrap();
    dict.set_item("status", format!("{:?}", info.info.status).to_lowercase())
        .unwrap();
    dict.set_item("created_at", info.info.created_at).unwrap();
    dict.set_item("updated_at", info.info.updated_at).unwrap();
    if let Some(parent) = info.info.parent_id {
        dict.set_item("parent_id", parent.as_str()).unwrap();
    }
    dict.set_item("version", info.version).unwrap();
    dict.set_item("timestamp", info.timestamp).unwrap();
    dict.unbind().into_any()
}

/// Convert BranchExportResult to Python dict.
fn branch_export_result_to_py(py: Python<'_>, r: BranchExportResult) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id).unwrap();
    dict.set_item("path", r.path).unwrap();
    dict.set_item("entry_count", r.entry_count).unwrap();
    dict.set_item("bundle_size", r.bundle_size).unwrap();
    dict.unbind().into_any()
}

/// Convert BranchImportResult to Python dict.
fn branch_import_result_to_py(py: Python<'_>, r: BranchImportResult) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id).unwrap();
    dict.set_item("transactions_applied", r.transactions_applied)
        .unwrap();
    dict.set_item("keys_written", r.keys_written).unwrap();
    dict.unbind().into_any()
}

/// Convert BundleValidateResult to Python dict.
fn bundle_validate_result_to_py(py: Python<'_>, r: BundleValidateResult) -> PyObject {
    let dict = PyDict::new_bound(py);
    dict.set_item("branch_id", r.branch_id).unwrap();
    dict.set_item("format_version", r.format_version).unwrap();
    dict.set_item("entry_count", r.entry_count).unwrap();
    dict.set_item("checksums_valid", r.checksums_valid).unwrap();
    dict.unbind().into_any()
}

/// Python module initialization.
#[pymodule]
fn _stratadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStrata>()?;
    Ok(())
}
