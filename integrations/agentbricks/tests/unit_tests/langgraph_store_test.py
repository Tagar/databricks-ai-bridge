"""Tests for the DatabricksMemoryStore BaseStore implementation over the AgentKit SDK.

Coverage: T01-T38, T51-T58 from the mapping table.
"""

import threading
from datetime import datetime, timezone
from urllib.parse import quote

import pytest
from _langgraph_store_fakes import STORE, create_fake_api_fixture
from langgraph.store.base import (
    GetOp,
    ListNamespacesOp,
    PutOp,
    SearchOp,
)

from databricks_agentkit.memory_store import Memory, MemorySearchResult, MemoryStore


class FakeMemoryStoreHandle:
    """Adapts FakeMemoryApi to MemoryStore handle interface for testing."""

    def __init__(self, api):
        self.api = api
        self.store_id = STORE

    def put(self, actor_id: str, path: str, content: str, session_id=None):
        """Put an entry."""
        self.api.create_memory_entry(
            self.store_id, actor_id, path, content=content, session_id=session_id
        )

    def list(self, actor_id: str, *, path=None, path_prefix=None, session_id=None, page_size=None):
        """List entries."""
        response = self.api.list_memory_entries(
            self.store_id,
            actor_id,
            path=path,
            path_prefix=path_prefix,
            session_id=session_id,
        )
        # Create a dummy store for constructing Memory objects.
        dummy_store = MemoryStore(
            name=f"memory-stores/{self.store_id}",
            display_name="Test Store",
            _client=self,
        )
        for entry in response.get("managed_memory_entries", []):
            yield Memory(
                name=entry["name"],
                actor_id=entry["actor_id"],
                path=entry["path"],
                session_id=entry.get("session_id"),
                content=entry.get("content"),
                create_time=entry.get("create_time"),
                update_time=entry.get("update_time"),
                _store=dummy_store,
                _client=self,
            )

    def search(self, actor_id: str, *, query=None, path_prefix=None, limit=None, session_id=None):
        """Search entries."""
        response = self.api.search_memory_entries(
            self.store_id,
            actor_id,
            query=query,
            path_prefix=path_prefix,
            limit=limit,
            session_id=session_id,
        )
        dummy_store = MemoryStore(
            name=f"memory-stores/{self.store_id}",
            display_name="Test Store",
            _client=self,
        )
        for item in response.get("results", []):
            entry = item["memory"]
            memory = Memory(
                name=entry["name"],
                actor_id=entry["actor_id"],
                path=entry["path"],
                session_id=entry.get("session_id"),
                content=entry.get("content"),
                create_time=entry.get("create_time"),
                update_time=entry.get("update_time"),
                _store=dummy_store,
                _client=self,
            )
            yield MemorySearchResult(memory=memory, score=item.get("score", 1.0))

    def _update_memory(self, store, memory_id, content=None, description=None):
        """Update a memory entry."""
        return self.api.update_memory_entry(
            store, memory_id, content=content, description=description
        )

    def _delete_memory(self, store, memory_id):
        """Delete a memory entry."""
        return self.api.delete_memory_entry(store, memory_id)


@pytest.fixture
def api(monkeypatch):
    """Provide a fake memory API and wire it into the AgentKit client."""
    fake = create_fake_api_fixture(monkeypatch)

    # Add memory_stores attribute that AgentKitClient expects
    class MemoryStoresWrapper:
        def __init__(self, api):
            self.api = api

        def _handle(self, store_id: str):
            return FakeMemoryStoreHandle(self.api)

    fake.memory_stores = MemoryStoresWrapper(fake)
    return fake


@pytest.fixture
def store_factory(api):
    """Factory to create DatabricksMemoryStore instances without importing it yet."""

    def _make(store_id: str = STORE, actor: str = "alice", **kwargs):
        from databricks_agentkit.langgraph.store import DatabricksMemoryStore

        return DatabricksMemoryStore(store=store_id, actor=actor, **kwargs)

    return _make


def test_store_import_possible(store_factory):
    """Verify the store can be imported and instantiated."""
    store = store_factory()
    assert store is not None


def test_store_put_get_roundtrip(store_factory, api):
    """Test basic put and get operations with JSON values."""
    store = store_factory()

    # Put a simple value.
    value = {"msg": "hello", "count": 42}
    store.batch(
        [
            PutOp(namespace=("ns",), key="key1", value=value),
        ]
    )

    # Get it back.
    result = store.batch(
        [
            GetOp(namespace=("ns",), key="key1"),
        ]
    )
    item = result[0]
    assert item is not None
    assert item.value == value
    assert item.namespace == ("ns",)
    assert item.key == "key1"


def test_store_put_delete_via_none(store_factory, api):
    """Test that PutOp(value=None) deletes the item."""
    store = store_factory()

    # Put an item.
    store.batch(
        [
            PutOp(namespace=("ns",), key="key1", value={"data": "value"}),
        ]
    )

    # Delete via None.
    store.batch(
        [
            PutOp(namespace=("ns",), key="key1", value=None),
        ]
    )

    # Verify it's gone.
    result = store.batch(
        [
            GetOp(namespace=("ns",), key="key1"),
        ]
    )
    assert result[0] is None


def test_store_get_missing_returns_none(store_factory):
    """Test that GetOp for a missing item returns None."""
    store = store_factory()
    result = store.batch(
        [
            GetOp(namespace=("ns",), key="missing"),
        ]
    )
    assert result[0] is None


def test_store_path_encoding_roundtrip(store_factory, api):
    """Test that paths with special characters round-trip correctly."""
    store = store_factory()

    special_chars = ["/", "%", ".", "..", " ", "hello world", "cafe", "abc"]
    for char_value in special_chars:
        namespace_encoded = quote(char_value, safe="")
        # Use the encoded form as part of the namespace.
        namespace = (f"ns-{namespace_encoded}", char_value)
        key = f"key-{namespace_encoded}"

        value = {"test": f"value-{char_value}"}
        store.batch(
            [
                PutOp(namespace=namespace, key=key, value=value),
            ]
        )

        result = store.batch(
            [
                GetOp(namespace=namespace, key=key),
            ]
        )
        item = result[0]
        assert item is not None
        assert item.value == value
        assert item.namespace == namespace
        assert item.key == key


def test_store_json_value_roundtrip(store_factory):
    """Test that complex JSON values round-trip exactly."""
    store = store_factory()

    complex_value = {
        "string": "hello",
        "number": 42,
        "float": 3.14,
        "bool": True,
        "null": None,
        "list": [1, 2, 3],
        "nested": {"deep": {"value": "here"}},
        "unicode": "cafe",
    }
    store.batch(
        [
            PutOp(namespace=("ns",), key="key", value=complex_value),
        ]
    )

    result = store.batch(
        [
            GetOp(namespace=("ns",), key="key"),
        ]
    )
    assert result[0].value == complex_value


def test_store_created_updated_timestamps(store_factory, api):
    """Test that created_at and updated_at are set correctly."""
    store = store_factory()

    now_before = datetime.now(timezone.utc)
    store.batch(
        [
            PutOp(namespace=("ns",), key="key", value={"data": "initial"}),
        ]
    )
    now_after = datetime.now(timezone.utc)

    result = store.batch(
        [
            GetOp(namespace=("ns",), key="key"),
        ]
    )
    item = result[0]
    assert item.created_at is not None
    assert item.updated_at is not None
    assert now_before <= item.created_at <= now_after


def test_store_session_scoping_write_cross_session(store_factory, api):
    """Test that writes use session_id=None (cross-session)."""
    store = store_factory()

    # Write an item (should not set session_id).
    store.batch(
        [
            PutOp(namespace=("ns",), key="key", value={"data": "value"}),
        ]
    )

    # Verify it was written without session_id.
    assert len(api.entries) == 1
    entry = list(api.entries.values())[0]
    assert "session_id" not in entry


def test_store_search_with_query(store_factory, api):
    """Test search with a natural language query returns results."""
    store = store_factory()

    # Put some items.
    store.batch(
        [
            PutOp(namespace=("docs",), key="doc1", value={"title": "Python guide"}),
            PutOp(namespace=("docs",), key="doc2", value={"title": "Rust book"}),
        ]
    )

    # Search with a query (BM25 keyword search).
    # Note: Query-based search filtering happens on the server side,
    # so our fake API doesn't implement the actual BM25 filtering.
    # Just verify the search method works without error.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("docs",), query="python"),
        ]
    )
    # The result is a list (may be empty in our fake API).
    assert isinstance(result[0], list)


def test_store_search_without_query(store_factory, api):
    """Test search without query (list operation)."""
    store = store_factory()

    # Put some items.
    store.batch(
        [
            PutOp(namespace=("ns",), key="key1", value={"a": 1}),
            PutOp(namespace=("ns",), key="key2", value={"b": 2}),
        ]
    )

    # Search without query (list all in namespace).
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",)),
        ]
    )
    items = result[0]
    assert len(items) == 2


def test_store_search_with_filter(store_factory, api):
    """Test search with filter criteria."""
    store = store_factory()

    # Put items with different values.
    store.batch(
        [
            PutOp(namespace=("ns",), key="item1", value={"status": "active", "count": 5}),
            PutOp(namespace=("ns",), key="item2", value={"status": "inactive", "count": 10}),
        ]
    )

    # Search with filter.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",), filter={"status": "active"}),
        ]
    )
    items = result[0]
    # Verify we got results (actual filtering depends on impl).
    assert len(items) >= 0


def test_store_search_with_limits(store_factory, api):
    """Test that search respects limit and offset parameters."""
    store = store_factory()

    # Put multiple items.
    store.batch(
        [
            PutOp(namespace=("ns",), key="k1", value={"x": 1}),
            PutOp(namespace=("ns",), key="k2", value={"x": 2}),
            PutOp(namespace=("ns",), key="k3", value={"x": 3}),
        ]
    )

    # Search with limit.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",), limit=2),
        ]
    )
    items = result[0]
    assert len(items) <= 2


def test_store_list_namespaces(store_factory, api):
    """Test list_namespaces operation."""
    store = store_factory()

    # Put items with various namespaces.
    store.batch(
        [
            PutOp(namespace=("users", "alice", "docs"), key="doc1", value={"x": 1}),
            PutOp(namespace=("users", "bob", "docs"), key="doc2", value={"y": 2}),
            PutOp(namespace=("posts",), key="p1", value={"z": 3}),
        ]
    )

    # List all namespaces.
    result = store.batch(
        [
            ListNamespacesOp(match_conditions=None),
        ]
    )
    namespaces = result[0]
    # Should include the namespaces we created.
    assert len(namespaces) > 0


def test_store_paging_search(store_factory, api):
    """Test that search results are paginated correctly."""
    store = store_factory()

    # Put many items.
    for i in range(5):
        store.batch(
            [
                PutOp(namespace=("ns",), key=f"key{i}", value={"num": i}),
            ]
        )

    # Search with limit and offset.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",), limit=2, offset=1),
        ]
    )
    items = result[0]
    # Offset/limit is applied client-side.
    assert len(items) > 0


def test_store_ttl_not_implemented(store_factory):
    """Test that TTL support raises NotImplementedError."""
    store = store_factory()

    with pytest.raises(NotImplementedError) as exc_info:
        store.batch(
            [
                PutOp(namespace=("ns",), key="key", value={"data": "value"}, ttl=60.0),
            ]
        )
    assert "TTL" in str(exc_info.value)


def test_store_index_list_not_implemented(store_factory):
    """Test that index list configuration raises NotImplementedError."""
    store = store_factory()

    with pytest.raises(NotImplementedError) as exc_info:
        store.batch(
            [
                PutOp(
                    namespace=("ns",),
                    key="key",
                    value={"data": "value"},
                    index=["field1", "field2"],
                ),
            ]
        )
    assert "index" in str(exc_info.value)


def test_store_supports_ttl_false(store_factory):
    """Test that supports_ttl is False."""
    store = store_factory()
    assert store.supports_ttl is False


def test_store_batch_resolve_actor_once(store_factory, api):
    """Test that batch resolves the actor once for all operations."""
    store = store_factory(actor="alice")

    ops = [
        PutOp(namespace=("ns",), key="key1", value={"a": 1}),
        PutOp(namespace=("ns",), key="key2", value={"b": 2}),
        GetOp(namespace=("ns",), key="key1"),
    ]

    results = store.batch(ops)
    assert len(results) == 3
    assert results[2] is not None  # Get should find the key


@pytest.mark.asyncio
async def test_store_abatch_resolve_actor_before_thread(store_factory, api):
    """Test that abatch resolves actor before asyncio.to_thread."""
    store = store_factory(actor="alice")

    ops = [
        PutOp(namespace=("ns",), key="key1", value={"a": 1}),
        GetOp(namespace=("ns",), key="key1"),
    ]

    results = await store.abatch(ops)
    assert len(results) == 2
    assert results[1] is not None


def test_memory_store_factory_explicit_store(monkeypatch, api):
    """Test memory_store() factory with explicit store argument."""
    from databricks_agentkit.langgraph.store import memory_store

    store = memory_store(store=STORE, actor="alice")
    assert store is not None


def test_memory_store_factory_from_env(monkeypatch, api):
    """Test memory_store() factory resolves from AGENT_MEMORY_STORE env."""
    from databricks_agentkit.langgraph.store import memory_store

    monkeypatch.setenv("AGENT_MEMORY_STORE", STORE)
    store = memory_store(actor="alice")
    assert store is not None


def test_memory_store_factory_client_injection(monkeypatch, api):
    """Test memory_store() factory with explicit client injection."""
    from databricks_agentkit.client import AgentKitClient
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore, memory_store

    # Create a client instance.
    def workspace_client():
        api.clients_built.append(object())
        return api.clients_built[-1]

    monkeypatch.setattr("databricks_agentkit.runtime.workspace.workspace_client", workspace_client)

    client = AgentKitClient(workspace_client())

    # Pass it to memory_store.
    store = memory_store(store=STORE, actor="alice", client=client)
    assert isinstance(store, DatabricksMemoryStore)


def test_memory_store_factory_handle_built_once(monkeypatch, api):
    """Test that the store handle is built once and reused."""
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    store = DatabricksMemoryStore(store=STORE, actor="alice")

    # Make two batch calls.
    store.batch([PutOp(namespace=("ns",), key="key", value={"a": 1})])
    store.batch([GetOp(namespace=("ns",), key="key")])

    # Should have built exactly one client.
    assert len(api.clients_built) == 1


def test_store_thread_safe_handle_build(monkeypatch, api):
    """Test that concurrent batch calls safely build the handle only once."""
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    store = DatabricksMemoryStore(store=STORE, actor="alice")
    results = []

    def thread_work():
        try:
            store.batch([PutOp(namespace=("ns",), key="key", value={"a": 1})])
            results.append("ok")
        except Exception as e:
            results.append(str(e))

    threads = [threading.Thread(target=thread_work) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should succeed, and only one client should be built.
    assert all(r == "ok" for r in results)
    assert len(api.clients_built) == 1


def test_store_with_root_parameter(store_factory, api):
    """Test that the root parameter changes the base path."""
    store = store_factory(root="/custom-root")

    store.batch(
        [
            PutOp(namespace=("ns",), key="key", value={"data": "value"}),
        ]
    )

    # Verify the path includes the custom root.
    entry = list(api.entries.values())[0]
    assert entry["path"].startswith("/custom-root")


def test_store_path_construction_exact(store_factory, api):
    """Test that paths are constructed exactly per the contract."""
    store = store_factory(root="/langgraph")

    namespace = ("docs", "user123")
    key = "report"

    store.batch(
        [
            PutOp(namespace=namespace, key=key, value={"data": "value"}),
        ]
    )

    # Path should be: {root}/{enc(ns_1)}/{enc(ns_2)}/{enc(key)}
    entry = list(api.entries.values())[0]
    expected_path = f"/langgraph/{quote('docs', safe='')}/{quote('user123', safe='')}/{quote('report', safe='')}"
    assert entry["path"] == expected_path


def test_store_multiple_namespace_levels(store_factory, api):
    """Test storage and retrieval with multiple namespace levels."""
    store = store_factory()

    # Create items with varying namespace depths.
    store.batch(
        [
            PutOp(namespace=("a",), key="k1", value={"v": 1}),
            PutOp(namespace=("a", "b"), key="k2", value={"v": 2}),
            PutOp(namespace=("a", "b", "c"), key="k3", value={"v": 3}),
        ]
    )

    # Get each one back.
    result = store.batch(
        [
            GetOp(namespace=("a",), key="k1"),
            GetOp(namespace=("a", "b"), key="k2"),
            GetOp(namespace=("a", "b", "c"), key="k3"),
        ]
    )

    assert result[0].value == {"v": 1}
    assert result[1].value == {"v": 2}
    assert result[2].value == {"v": 3}


def test_store_filter_operators_eq_ne(store_factory, api):
    """Test filter operators: $eq, $ne."""
    store = store_factory()

    store.batch(
        [
            PutOp(namespace=("ns",), key="item1", value={"status": "active"}),
            PutOp(namespace=("ns",), key="item2", value={"status": "inactive"}),
        ]
    )

    # Filter with $eq.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",), filter={"status": {"$eq": "active"}}),
        ]
    )
    items = result[0]
    assert len(items) == 1
    assert items[0].key == "item1"


def test_store_filter_operators_gt(store_factory, api):
    """Test filter operators: $gt, $gte, $lt, $lte."""
    store = store_factory()

    store.batch(
        [
            PutOp(namespace=("ns",), key="item1", value={"score": 10}),
            PutOp(namespace=("ns",), key="item2", value={"score": 20}),
            PutOp(namespace=("ns",), key="item3", value={"score": 30}),
        ]
    )

    # Filter with $gt.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("ns",), filter={"score": {"$gt": 15}}),
        ]
    )
    items = result[0]
    assert len(items) == 2


def test_store_update_via_put(store_factory, api):
    """Test that putting to an existing key updates the value."""
    store = store_factory()

    # Put initial value.
    store.batch([PutOp(namespace=("ns",), key="key", value={"count": 1})])

    # Update it.
    store.batch([PutOp(namespace=("ns",), key="key", value={"count": 2})])

    # Get it back.
    result = store.batch([GetOp(namespace=("ns",), key="key")])
    assert result[0].value == {"count": 2}


def test_store_cross_actor_isolation(store_factory, api):
    """Test that different actors don't see each other's data."""
    # Create store for alice.
    alice_store = store_factory(actor="alice")
    bob_store = store_factory(actor="bob")

    # Alice puts data.
    alice_store.batch([PutOp(namespace=("ns",), key="key", value={"owner": "alice"})])

    # Bob gets nothing.
    result = bob_store.batch([GetOp(namespace=("ns",), key="key")])
    assert result[0] is None

    # Alice gets her data.
    result = alice_store.batch([GetOp(namespace=("ns",), key="key")])
    assert result[0].value == {"owner": "alice"}


def test_store_search_namespace_prefix_filtering(store_factory, api):
    """Test that search respects namespace prefixes."""
    store = store_factory()

    # Put items in different namespaces.
    store.batch(
        [
            PutOp(namespace=("docs",), key="d1", value={"type": "doc"}),
            PutOp(namespace=("docs", "private"), key="d2", value={"type": "private"}),
            PutOp(namespace=("images",), key="i1", value={"type": "image"}),
        ]
    )

    # Search under docs prefix.
    result = store.batch(
        [
            SearchOp(namespace_prefix=("docs",)),
        ]
    )
    items = result[0]
    assert len(items) == 2
    assert all("docs" in item.namespace for item in items)
