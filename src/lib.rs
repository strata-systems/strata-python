//! Python bindings for StrataDB.
//!
//! This module exposes the StrataDB API to Python via PyO3.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use ::stratadb::{
    CollectionInfo, DistanceMetric, Error as StrataError, MergeStrategy,
    Strata as RustStrata, Value, VersionedValue,
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
}

#[pymethods]
impl PyStrata {
    /// Open a database at the given path.
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let inner = RustStrata::open(path).map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Create an in-memory database (no persistence).
    #[staticmethod]
    fn cache() -> PyResult<Self> {
        let inner = RustStrata::cache().map_err(to_py_err)?;
        Ok(Self { inner })
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
    fn state_cas(
        &self,
        cell: &str,
        new_value: &Bound<'_, PyAny>,
        expected_version: Option<u64>,
    ) -> PyResult<Option<u64>> {
        let new = py_to_value(new_value)?;
        self.inner.state_cas(cell, expected_version, new).map_err(to_py_err)
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

    /// List JSON document keys.
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

/// Python module initialization.
#[pymodule]
fn _stratadb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStrata>()?;
    Ok(())
}
