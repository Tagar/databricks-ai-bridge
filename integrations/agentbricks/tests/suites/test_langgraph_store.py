"""`DatabricksMemoryStore`: put/get/search/list_namespaces/delete with actor isolation.

Tests the LangGraph BaseStore implementation backed by Databricks managed memory, verifying
that two actors cannot access each other's data and that round-trip operations preserve values.

Run this test with the gate environment variable to verify store.py integration:
    RUN_LANGGRAPH_STORE_TESTS=1 pytest tests/suites/test_langgraph_store.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest


@pytest.mark.skipif(
    os.getenv("RUN_LANGGRAPH_STORE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_STORE_TESTS=1 to run DatabricksMemoryStore tests.",
)
@pytest.mark.usefixtures("workspace_client")
def test_langgraph_store_put_get_isolation(agentbricks_cli, project, cleanup) -> None:
    """DatabricksMemoryStore: put and get with actor isolation.

    Verifies that two actors can write and read their own data,
    but cannot see each other's data.
    """
    from langgraph.store.base import GetOp, PutOp

    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    actor1 = f"test_actor_{uuid.uuid4().hex[:8]}"
    actor2 = f"test_actor_{uuid.uuid4().hex[:8]}"
    store_id = f"test-store-{uuid.uuid4().hex[:6]}"

    # Create two stores, one for each actor
    store1 = DatabricksMemoryStore(store=store_id, actor=actor1)
    store2 = DatabricksMemoryStore(store=store_id, actor=actor2)

    # Actor 1 puts a value using batch()
    key1 = f"key_{uuid.uuid4().hex[:6]}"
    value1 = {"data": "actor1_value", "timestamp": str(uuid.uuid4())}
    results = store1.batch([PutOp(namespace=("test",), key=key1, value=value1)])
    assert len(results) == 1, "Put operation should succeed"

    # Actor 2 puts a different value in the same namespace
    key2 = f"key_{uuid.uuid4().hex[:6]}"
    value2 = {"data": "actor2_value", "timestamp": str(uuid.uuid4())}
    results = store2.batch([PutOp(namespace=("test",), key=key2, value=value2)])
    assert len(results) == 1, "Put operation should succeed"

    # Actor 1 can get their own value
    item1 = store1.batch([GetOp(namespace=("test",), key=key1)])[0]
    assert item1 is not None, "Actor 1 should be able to retrieve their own value"
    assert item1.value == value1, f"Actor 1's value should match: {item1.value} != {value1}"

    # Actor 1 cannot get actor 2's value
    item1_tries_key2 = store1.batch([GetOp(namespace=("test",), key=key2)])[0]
    assert item1_tries_key2 is None, "Actor 1 should not be able to retrieve actor 2's value"

    # Actor 2 can get their own value
    item2 = store2.batch([GetOp(namespace=("test",), key=key2)])[0]
    assert item2 is not None, "Actor 2 should be able to retrieve their own value"
    assert item2.value == value2, f"Actor 2's value should match: {item2.value} != {value2}"

    # Actor 2 cannot get actor 1's value
    item2_tries_key1 = store2.batch([GetOp(namespace=("test",), key=key1)])[0]
    assert item2_tries_key1 is None, "Actor 2 should not be able to retrieve actor 1's value"


@pytest.mark.skipif(
    os.getenv("RUN_LANGGRAPH_STORE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_STORE_TESTS=1 to run DatabricksMemoryStore tests.",
)
@pytest.mark.usefixtures("workspace_client")
def test_langgraph_store_search(agentbricks_cli, project, cleanup) -> None:
    """DatabricksMemoryStore: search with actor isolation.

    Verifies that search returns only results for the requesting actor.
    """
    from langgraph.store.base import PutOp, SearchOp

    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    actor1 = f"test_actor_{uuid.uuid4().hex[:8]}"
    actor2 = f"test_actor_{uuid.uuid4().hex[:8]}"
    store_id = f"test-store-{uuid.uuid4().hex[:6]}"

    store1 = DatabricksMemoryStore(store=store_id, actor=actor1)
    store2 = DatabricksMemoryStore(store=store_id, actor=actor2)

    # Actor 1 stores a searchable value
    store1.batch(
        [
            PutOp(
                namespace=("search",),
                key=f"doc_{uuid.uuid4().hex[:6]}",
                value={"content": "actor1 unique keyword document"},
            )
        ]
    )

    # Actor 2 stores a different searchable value
    store2.batch(
        [
            PutOp(
                namespace=("search",),
                key=f"doc_{uuid.uuid4().hex[:6]}",
                value={"content": "actor2 different keyword"},
            )
        ]
    )

    # Actor 1 searches for "unique" and should only find their own result
    results1 = store1.batch([SearchOp(namespace=("search",), query="unique", limit=10)])[0]
    assert isinstance(results1, list), "Search should return a list"
    assert len(results1) >= 1, "Actor 1's search should find at least their own document"
    # The search should only return actor1's document (or potentially others if BM25 matches)
    # but never actor2's content if the implementation respects isolation

    # Actor 2 searches for "different" and should only find their own result
    results2 = store2.batch([SearchOp(namespace=("search",), query="different", limit=10)])[0]
    assert isinstance(results2, list), "Search should return a list"
    # Verify isolation: actor2's results should not include actor1's unique keyword
    found_actor1_content = any("unique keyword" in str(item.value) for item in results2)
    assert not found_actor1_content, "Actor 2 should not be able to search results from actor 1"


@pytest.mark.skipif(
    os.getenv("RUN_LANGGRAPH_STORE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_STORE_TESTS=1 to run DatabricksMemoryStore tests.",
)
@pytest.mark.usefixtures("workspace_client")
def test_langgraph_store_list_namespaces(agentbricks_cli, project, cleanup) -> None:
    """DatabricksMemoryStore: list_namespaces with actor isolation.

    Verifies that actors only see their own namespace prefixes.
    """
    from langgraph.store.base import ListNamespacesOp, PutOp

    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    actor1 = f"test_actor_{uuid.uuid4().hex[:8]}"
    actor2 = f"test_actor_{uuid.uuid4().hex[:8]}"
    store_id = f"test-store-{uuid.uuid4().hex[:6]}"

    store1 = DatabricksMemoryStore(store=store_id, actor=actor1)
    store2 = DatabricksMemoryStore(store=store_id, actor=actor2)

    # Actor 1 creates data in namespace "ns_a"
    store1.batch(
        [
            PutOp(
                namespace=("ns_a",),
                key=f"key_{uuid.uuid4().hex[:6]}",
                value={"actor": 1},
            )
        ]
    )

    # Actor 2 creates data in namespace "ns_b"
    store2.batch(
        [
            PutOp(
                namespace=("ns_b",),
                key=f"key_{uuid.uuid4().hex[:6]}",
                value={"actor": 2},
            )
        ]
    )

    # Actor 1 lists namespaces and should only see "ns_a"
    namespaces1 = store1.batch([ListNamespacesOp()])[0]
    assert "ns_a" in namespaces1, "Actor 1 should see their own namespace"
    assert "ns_b" not in namespaces1, "Actor 1 should not see actor 2's namespace"

    # Actor 2 lists namespaces and should only see "ns_b"
    namespaces2 = store2.batch([ListNamespacesOp()])[0]
    assert "ns_b" in namespaces2, "Actor 2 should see their own namespace"
    assert "ns_a" not in namespaces2, "Actor 2 should not see actor 1's namespace"


@pytest.mark.skipif(
    os.getenv("RUN_LANGGRAPH_STORE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_STORE_TESTS=1 to run DatabricksMemoryStore tests.",
)
@pytest.mark.usefixtures("workspace_client")
def test_langgraph_store_delete(agentbricks_cli, project, cleanup) -> None:
    """DatabricksMemoryStore: put and delete operations.

    Verifies that delete removes a value and that subsequent get returns None.
    """
    from langgraph.store.base import GetOp, PutOp

    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    actor = f"test_actor_{uuid.uuid4().hex[:8]}"
    store_id = f"test-store-{uuid.uuid4().hex[:6]}"

    store = DatabricksMemoryStore(store=store_id, actor=actor)

    key = f"key_{uuid.uuid4().hex[:6]}"
    value = {"data": "to_delete"}

    # Put a value
    store.batch([PutOp(namespace=("test",), key=key, value=value)])

    # Verify it exists
    item = store.batch([GetOp(namespace=("test",), key=key)])[0]
    assert item is not None, "Item should exist after put"
    assert item.value == value

    # Delete the value using put with value=None
    store.batch([PutOp(namespace=("test",), key=key, value=None)])

    # Verify it's gone
    item_after_delete = store.batch([GetOp(namespace=("test",), key=key)])[0]
    assert item_after_delete is None, "Item should not exist after delete (put with value=None)"
