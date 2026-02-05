"""
StrataDB Python SDK

An embedded database for AI agents with six primitives: KV Store, Event Log,
State Cell, JSON Store, Vector Store, and Branches.

Example usage:

    from stratadb import Strata

    db = Strata.open("/data")
    db.kv_put("user:123", "Alice")

    # Branch isolation
    db.create_branch("experiment")
    db.set_branch("experiment")

    # Vector search with NumPy
    import numpy as np
    embedding = np.random.rand(384).astype(np.float32)
    db.vector_create_collection("docs", 384)
    db.vector_upsert("docs", "doc-1", embedding)
    results = db.vector_search("docs", embedding, k=5)
"""

from ._stratadb import Strata

__all__ = ["Strata"]
__version__ = "0.6.0"
