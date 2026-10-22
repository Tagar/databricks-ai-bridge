"""Live check of ``DatabricksMemoryStore`` against a real managed memory store (gated).

Each test creates its own memory store and registers its deletion before the create, then drives
the public LangGraph store methods. The store uses the suite's workspace client, not ambient auth.

Run with the gate set:
    RUN_LANGGRAPH_STORE_TESTS=1 pytest tests/suites/test_langgraph_store.py -v
"""

from __future__ import annotations

import os
import uuid

import pytest

gated = pytest.mark.skipif(
    os.getenv("RUN_LANGGRAPH_STORE_TESTS") != "1",
    reason="Set RUN_LANGGRAPH_STORE_TESTS=1 to run DatabricksMemoryStore tests.",
)


def _new_store(agentbricks_cli, cleanup) -> str:
    """Create a memory store and return its id. The create takes the display name as the id."""
    store_id = f"lg-store-{uuid.uuid4().hex[:10]}"
    # Registered first: a create that fails partway can still leave the store behind.
    cleanup(f"memory stores delete memory-stores/{store_id} --yes")
    agentbricks_cli.run(f"memory stores create --display-name {store_id}")
    return store_id


@gated
def test_langgraph_store_put_get_overwrite_delete(workspace_client, agentbricks_cli, cleanup):
    """DatabricksMemoryStore: put, get, overwrite, search and delete on a live store."""
    from langgraph.store.base import PutOp

    from databricks_agentkit.client import AgentKitClient
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    store_id = _new_store(agentbricks_cli, cleanup)
    store = DatabricksMemoryStore(
        store=store_id,
        actor=f"e2e-{uuid.uuid4().hex[:8]}",
        client=AgentKitClient(workspace_client),
    )

    store.put(("notes",), "k1", {"text": "alpha"})
    item = store.get(("notes",), "k1")
    assert item is not None and item.value == {"text": "alpha"}

    store.put(("notes",), "k1", {"text": "beta"})  # the second write takes the update path
    item = store.get(("notes",), "k1")
    assert item is not None and item.value == {"text": "beta"}
    assert [hit.key for hit in store.search(("notes",))] == ["k1"]

    store.batch([PutOp(namespace=("notes",), key="k1", value=None)])  # a None value deletes
    assert store.get(("notes",), "k1") is None


@gated
def test_langgraph_store_isolates_actors(workspace_client, agentbricks_cli, cleanup):
    """DatabricksMemoryStore: two actors on one live store see only their own entries."""
    from databricks_agentkit.client import AgentKitClient
    from databricks_agentkit.langgraph.store import DatabricksMemoryStore

    store_id = _new_store(agentbricks_cli, cleanup)
    client = AgentKitClient(workspace_client)
    suffix = uuid.uuid4().hex[:8]
    alice = DatabricksMemoryStore(store=store_id, actor=f"alice-{suffix}", client=client)
    bob = DatabricksMemoryStore(store=store_id, actor=f"bob-{suffix}", client=client)

    # The same key in the same namespace for both actors: no collision, each keeps its own value.
    alice.put(("notes",), "shared-key", {"owner": "alice"})
    bob.put(("notes",), "shared-key", {"owner": "bob"})
    bob.put(("bob-only",), "secret", {"owner": "bob"})

    alice_item = alice.get(("notes",), "shared-key")
    bob_item = bob.get(("notes",), "shared-key")
    assert alice_item is not None and alice_item.value == {"owner": "alice"}
    assert bob_item is not None and bob_item.value == {"owner": "bob"}
    assert alice.get(("bob-only",), "secret") is None
    assert [hit.key for hit in alice.search(("notes",))] == ["shared-key"]
    assert alice.list_namespaces() == [("notes",)]
